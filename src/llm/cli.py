import argparse,json
from pathlib import Path
import torch
from .config import Config
from .model import LanguageModel,parameter_report
from .tokenizer import BPETokenizer
from .data import documents,prepare,TokenData
from .training import train,evaluate,precision
from .generation import load_model,generate

def main():
    p=argparse.ArgumentParser(description='Train a local decoder-only language model')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('count-parameters'); q.add_argument('--config',default='configs/130m.yaml')
    q=sub.add_parser('train-tokenizer'); q.add_argument('--input',required=True); q.add_argument('--output',default='data/tokenizer/tokenizer.json'); q.add_argument('--vocab-size',type=int,default=50257)
    q=sub.add_parser('prepare-data'); q.add_argument('--input',required=True); q.add_argument('--tokenizer',required=True); q.add_argument('--output',default='data/processed'); q.add_argument('--seed',type=int,default=42)
    for name in ('train','finetune'):
        q=sub.add_parser(name); q.add_argument('--config',default='configs/laptop.yaml'); q.add_argument('--tokenizer',required=True); q.add_argument('--data',required=True); q.add_argument('--output',default='checkpoints'); q.add_argument('--resume',action='store_true'); q.add_argument('--stop-after',type=int)
        if name=='finetune':
            q.add_argument('--checkpoint',required=True); q.add_argument('--instructions',required=True); q.add_argument('--template'); q.add_argument('--all-tokens',action='store_true')
    for name in ('evaluate','generate','serve','export'):
        q=sub.add_parser(name); q.add_argument('--checkpoint',required=True); q.add_argument('--tokenizer')
        if name=='evaluate': q.add_argument('--data',required=True)
        if name=='generate':
            q.add_argument('--prompt',required=True); q.add_argument('--max-new-tokens',type=int,default=100); q.add_argument('--temperature',type=float,default=0.8); q.add_argument('--top-k',type=int,default=50); q.add_argument('--top-p',type=float,default=0.95); q.add_argument('--repetition-penalty',type=float,default=1.0)
        if name=='serve': q.add_argument('--port',type=int,default=8000)
        if name=='export': q.add_argument('--output',default='export')
    a=p.parse_args()
    try:
        if a.command=='count-parameters':
            c=Config.load(a.config)
            with torch.device('meta'): m=LanguageModel(c.model)
            print(json.dumps(parameter_report(m),indent=2)); return
        if a.command=='train-tokenizer':
            tok=BPETokenizer.train(documents(a.input),a.output,a.vocab_size); print('Actual vocabulary:',tok.vocab_size); return
        if a.command=='prepare-data':
            print(json.dumps(prepare(a.input,BPETokenizer.load(a.tokenizer),a.output,seed=a.seed),indent=2)); return
        if a.command in ('train','finetune'):
            if a.stop_after is not None and a.stop_after<1: p.error('--stop-after must be positive')
            opts={}
            if a.command=='finetune': opts={'init_checkpoint':a.checkpoint,'sft':a.instructions,'template':Path(a.template).read_text() if a.template else None,'response_only':not a.all_tokens}
            train(Config.load(a.config),BPETokenizer.load(a.tokenizer),a.data,a.output,a.resume,stop_after=a.stop_after,**opts); return
        model,tok,c=load_model(a.checkpoint,a.tokenizer)
        if a.command=='generate':
            print(json.dumps(generate(model,tok,a.prompt,a.max_new_tokens,a.temperature,a.top_k,a.top_p,a.repetition_penalty),ensure_ascii=False,indent=2))
        elif a.command=='evaluate':
            meta=json.loads((Path(a.data).parent/'metadata.json').read_text())
            if meta['tokenizer_hash']!=tok.fingerprint: raise ValueError('Evaluation tokenizer mismatch')
            device=next(model.parameters()).device
            print(json.dumps(evaluate(model,TokenData(a.data,c.training.sequence_length),c.training,device,precision(device,c.training.precision)),indent=2))
        elif a.command=='serve':
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(model,tok),host='127.0.0.1',port=a.port)
        elif a.command=='export':
            from safetensors.torch import save_file
            out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(out/'model.safetensors'))
            (out/'config.json').write_text(json.dumps(c.dict(),indent=2)); tok.save(out/'tokenizer.json')
            print('Exported weights/config/tokenizer; load with this package, not Hugging Face AutoModel')
    except (ValueError,FileNotFoundError,FloatingPointError) as e: p.exit(1,f'Error: {e}\n')
    except torch.cuda.OutOfMemoryError: p.exit(1,'GPU memory exhausted. Reduce batch_size or sequence_length in a NEW run; enable gradient_checkpointing.\n')

if __name__=='__main__': main()
