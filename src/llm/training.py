import contextlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import LanguageModel
from .data import TokenData,InstructionData

def device_info():
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0))) if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    return device,{'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'vram_bytes':torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None,'gpu_count':torch.cuda.device_count(),'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda}

def precision(device,mode):
    if device.type!='cuda': return torch.float32
    if mode=='fp32': return torch.float32
    if mode in ('auto','bf16') and torch.cuda.is_bf16_supported(): return torch.bfloat16
    return torch.float16

def amp(device,dtype): return torch.autocast(device.type,dtype=dtype) if dtype!=torch.float32 else contextlib.nullcontext()

def lr_at(step,t):
    if step<t.warmup_steps: return t.learning_rate*(step+1)/t.warmup_steps
    ratio=min(1,(step-t.warmup_steps)/max(1,t.max_steps-1-t.warmup_steps))
    return t.min_lr+(t.learning_rate-t.min_lr)*0.5*(1+math.cos(math.pi*ratio))

def optimizer_for(model,t):
    groups=[{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':t.weight_decay},{'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.0}]
    return torch.optim.AdamW(groups,lr=t.learning_rate,betas=(t.beta1,t.beta2),eps=1e-8)

def rng_state(generator):
    s=np.random.get_state()
    return {'python':random.getstate(),'numpy':[s[0],s[1].tolist(),s[2],s[3],s[4]],'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'data':generator.get_state()}

def restore_rng(state,generator):
    random.setstate(state['python']); s=state['numpy']; np.random.set_state((s[0],np.array(s[1],dtype=np.uint32),s[2],s[3],s[4]))
    torch.set_rng_state(state['torch']); generator.set_state(state['data'])
    if state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])

def load_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format_version')!=1: raise ValueError('Unsupported checkpoint format')
    return saved

@torch.no_grad()
def evaluate(model,data,t,device,dtype):
    was_training=model.training; model.eval(); g=torch.Generator().manual_seed(9173)
    total=torch.zeros(2,device=device,dtype=torch.float64)
    for _ in range(t.eval_batches):
        x,y=data.batch(t.batch_size,g,device)
        with amp(device,dtype): _,loss=model(x,y)
        count=(y!=-100).sum(); total[0]+=loss.double()*count; total[1]+=count
    if dist.is_initialized(): dist.all_reduce(total)
    model.train(was_training)
    value=(total[0]/total[1]).item()
    if not math.isfinite(value): raise FloatingPointError('Nonfinite validation loss')
    return {'validation_loss':value,'perplexity':math.exp(value) if value<700 else None}

def train(config,tokenizer,data_dir,out,resume=False,init_checkpoint=None,sft=None,template=None,response_only=True,stop_after=None):
    world=int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('RANK','0'))
    device,info=device_info()
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group(backend='nccl' if device.type=='cuda' else 'gloo')
    t=config.training; dtype=precision(device,t.precision); info['precision']=str(dtype)
    random.seed(t.seed+rank); np.random.seed(t.seed+rank); torch.manual_seed(t.seed+rank)
    torch.use_deterministic_algorithms(t.deterministic)
    if tokenizer.vocab_size>config.model.vocab_size: raise ValueError('Tokenizer exceeds model vocabulary')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if not resume and (out/'latest.pt').exists(): raise ValueError('Output already has a checkpoint; use --resume or choose a new output folder')
    saved=load_checkpoint(out/'latest.pt') if resume else load_checkpoint(init_checkpoint) if init_checkpoint else None
    model=LanguageModel(config.model).to(device)
    if saved:
        if saved['config']['model']!=config.dict()['model']: raise ValueError('Checkpoint model config mismatch')
        if saved['tokenizer_hash']!=tokenizer.fingerprint: raise ValueError('Checkpoint tokenizer mismatch')
        model.load_state_dict(saved['model'])
    data_dir=Path(data_dir)
    if sft:
        opts={} if template is None else {'template':template}
        train_data=InstructionData(sft,tokenizer,t.sequence_length,rank,world,response_only=response_only,**opts)
        val_data=InstructionData(data_dir/'validation.jsonl',tokenizer,t.sequence_length,rank,world,response_only=response_only,**opts)
        import hashlib
        signature={'sft':hashlib.sha256(Path(sft).read_bytes()).hexdigest(),'validation':hashlib.sha256((data_dir/'validation.jsonl').read_bytes()).hexdigest(),'template':template,'response_only':response_only}
    else:
        signature=json.loads((data_dir/'metadata.json').read_text())
        if signature['tokenizer_hash']!=tokenizer.fingerprint: raise ValueError('Dataset tokenizer mismatch')
        train_data=TokenData(data_dir/'train.bin',t.sequence_length,rank,world)
        val_data=TokenData(data_dir/'validation.bin',t.sequence_length,rank,world)
    optim=optimizer_for(model,t); scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
    g=torch.Generator().manual_seed(t.seed+rank); start=0
    if resume:
        if saved['config']!=config.dict() or saved['world_size']!=world: raise ValueError('Exact resume requires unchanged config and world size')
        if saved['data_signature']!=signature: raise ValueError('Dataset changed since checkpoint')
        optim.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); start=saved['step']
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None) if world>1 else model
    if resume: restore_rng(saved['rng'][rank],g)
    if rank==0: print(json.dumps(info),flush=True)
    end=min(t.max_steps,start+stop_after) if stop_after else t.max_steps
    since=time.monotonic()
    for step in range(start,end):
        model.train(); optim.zero_grad(set_to_none=True)
        for group in optim.param_groups: group['lr']=lr_at(step,t)
        loss_total=0.0
        for micro in range(t.gradient_accumulation_steps):
            sync=wrapped.no_sync() if world>1 and micro<t.gradient_accumulation_steps-1 else contextlib.nullcontext()
            with sync:
                x,y=train_data.batch(t.batch_size,g,device)
                with amp(device,dtype): _,loss=wrapped(x,y)
                finite=torch.tensor(int(torch.isfinite(loss)),device=device)
                if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item(): raise FloatingPointError('Nonfinite loss: inspect data, learning rate and precision; last checkpoint is preserved')
                loss_total+=loss.item(); scaler.scale(loss/t.gradient_accumulation_steps).backward()
        scaler.unscale_(optim)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        scaler.step(optim); scaler.update()
        row={'step':step+1,'train_loss':loss_total/t.gradient_accumulation_steps,'learning_rate':optim.param_groups[0]['lr'],'gradient_norm':float(norm)}
        if (step+1)%t.eval_interval==0 or step+1==end: row.update(evaluate(model,val_data,t,device,dtype))
        if rank==0 and ((step+1)%t.log_interval==0 or step+1==end):
            elapsed=time.monotonic()-since
            row.update(elapsed_seconds=elapsed,tokens_per_second=(step+1-start)*t.batch_size*t.gradient_accumulation_steps*t.sequence_length*world/elapsed,gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
            print(json.dumps(row),flush=True)
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        if (step+1)%t.checkpoint_interval==0 or step+1==end:
            local=rng_state(g); states=[None]*world if rank==0 else None
            if world>1: dist.gather_object(local,states,dst=0)
            else: states=[local]
            if rank==0:
                payload={'format_version':1,'model':model.state_dict(),'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),'scheduler':{'step':step+1},'step':step+1,'epoch':None,'config':config.dict(),'tokenizer_hash':tokenizer.fingerprint,'data_signature':signature,'world_size':world,'rng':states}
                torch.save(payload,out/'pending.pt'); (out/'pending.pt').replace(out/'latest.pt')
                tokenizer.save(out/'tokenizer.json')
            if world>1: dist.barrier()
    if world>1: dist.destroy_process_group()
    return model
