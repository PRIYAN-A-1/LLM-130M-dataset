import torch
from .config import Config
from .model import LanguageModel
from .tokenizer import BPETokenizer
from .training import load_checkpoint,device_info

def sample(logits,temperature=0.8,top_k=50,top_p=0.95):
    if temperature<0 or top_k<0 or not 0<top_p<=1: raise ValueError('Invalid sampling options')
    if temperature==0: return logits.argmax(-1,keepdim=True)
    logits=logits.float()/temperature
    if top_k:
        limit=logits.topk(min(top_k,logits.shape[-1])).values[:,-1:]
        logits=logits.masked_fill(logits<limit,float('-inf'))
    if top_p<1:
        sorted_logits,indices=logits.sort(descending=True)
        remove=sorted_logits.softmax(-1).cumsum(-1)>top_p
        remove[:,1:]=remove[:,:-1].clone(); remove[:,0]=False
        logits=logits.scatter(1,indices,sorted_logits.masked_fill(remove,float('-inf')))
    return torch.multinomial(logits.softmax(-1),1)

@torch.inference_mode()
def generate(model,tokenizer,prompt,max_new_tokens=100,temperature=0.8,top_k=50,top_p=0.95,repetition_penalty=1.0,stop_tokens=None):
    if not 1<=max_new_tokens<=2048 or repetition_penalty<=0: raise ValueError('Invalid generation limits')
    if temperature<0 or top_k<0 or not 0<top_p<=1: raise ValueError('Invalid sampling options')
    device=next(model.parameters()).device; model.eval()
    initial=tokenizer.encode(prompt) or [tokenizer.bos_id]
    ids=torch.tensor([initial],device=device); produced=[]
    stops=set(stop_tokens if stop_tokens is not None else [tokenizer.eos_id])
    for _ in range(max_new_tokens):
        logits,_=model(ids[:,-model.config.context_length:]); logits=logits[:,-1,:].float()
        logits[:,tokenizer.vocab_size:]=float('-inf')
        seen=torch.unique(ids)
        if repetition_penalty!=1:
            values=logits[:,seen]; logits[:,seen]=torch.where(values<0,values*repetition_penalty,values/repetition_penalty)
        nxt=sample(logits,temperature,top_k,top_p); token=nxt.item()
        produced.append(token); ids=torch.cat([ids,nxt],dim=1)
        if token in stops: break
    return {'text':tokenizer.decode(initial+produced),'completion':tokenizer.decode(produced),'tokens_generated':len(produced)}

def load_model(path,tokenizer_path=None):
    from pathlib import Path
    saved=load_checkpoint(path); config=Config.from_dict(saved['config'])
    tok=BPETokenizer.load(tokenizer_path or Path(path).parent/'tokenizer.json')
    if tok.fingerprint!=saved['tokenizer_hash']: raise ValueError('Tokenizer does not match checkpoint')
    device,_=device_info(); model=LanguageModel(config.model).to(device)
    model.load_state_dict(saved['model']); model.eval(); return model,tok,config
