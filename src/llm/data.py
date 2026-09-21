import hashlib,json
from pathlib import Path
import numpy as np
import torch

def documents(source):
    root=Path(source)
    paths=sorted(root.rglob('*')) if root.is_dir() else [root]
    found=False
    for path in paths:
        if path.suffix.lower()=='.txt':
            found=True
            with path.open(encoding='utf-8') as f:
                # Each nonempty line is a document; never read the full corpus into RAM.
                for line in f:
                    if line.strip(): yield line.rstrip('\r\n')
        elif path.suffix.lower()=='.jsonl':
            found=True
            with path.open(encoding='utf-8') as f:
                for i,line in enumerate(f,1):
                    if not line.strip(): continue
                    row=json.loads(line)
                    if not isinstance(row.get('text'),str): raise ValueError(f'{path}:{i}: expected text string')
                    if row['text'].strip(): yield row['text']
    if not found: raise ValueError('No .txt or .jsonl input files found')

def prepare(source,tokenizer,out,val_fraction=0.1,seed=42):
    if not 0<val_fraction<1: raise ValueError('val_fraction must be between zero and one')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    counts={'train':0,'validation':0}; hashes={k:hashlib.sha256() for k in counts}
    handles={k:(out/f'{k}.bin').open('wb') for k in counts}
    try:
        for doc in documents(source):
            digest=hashlib.sha256((str(seed)+'\0'+doc).encode()).digest()
            split='validation' if int.from_bytes(digest[:8],'big')/2**64<val_fraction else 'train'
            raw=np.asarray(tokenizer.encode(doc)+[tokenizer.eos_id],dtype='<u4').tobytes()
            handles[split].write(raw); counts[split]+=len(raw)//4; hashes[split].update(raw)
    finally:
        for f in handles.values(): f.close()
    if not all(counts.values()): raise ValueError('Empty split: provide more distinct documents or change seed')
    meta={'counts':counts,'tokenizer_hash':tokenizer.fingerprint,'vocab_size':tokenizer.vocab_size,'sha256':{k:h.hexdigest() for k,h in hashes.items()},'dtype':'uint32-little-endian','seed':seed}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)); return meta

class TokenData:
    def __init__(self,path,seq,rank=0,world=1):
        if Path(path).stat().st_size%4: raise ValueError('Token file is not uint32 aligned')
        self.tokens=np.memmap(path,dtype='<u4',mode='r'); self.seq=seq
        self.lo=len(self.tokens)*rank//world; self.hi=len(self.tokens)*(rank+1)//world
        if self.hi-self.lo<=seq: raise ValueError(f'{path}: each rank needs more than {seq} tokens')
    def batch(self,batch_size,generator,device):
        starts=torch.randint(self.lo,self.hi-self.seq,(batch_size,),generator=generator).tolist()
        a=torch.from_numpy(np.stack([self.tokens[s:s+self.seq+1] for s in starts]).astype(np.int64))
        if device.type=='cuda': a=a.pin_memory()
        a=a.to(device,non_blocking=True)
        return a[:,:-1],a[:,1:]

DEFAULT_TEMPLATE='### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n'
class InstructionData:
    def __init__(self,path,tokenizer,seq,rank=0,world=1,template=DEFAULT_TEMPLATE,response_only=True):
        self.rows=[]; self.seq=seq; self.pad=tokenizer.backend.token_to_id('<PAD>')
        with open(path,encoding='utf-8') as f:
            for i,line in enumerate(f):
                if not line.strip() or i%world!=rank: continue
                row=json.loads(line)
                if not isinstance(row.get('instruction'),str) or not isinstance(row.get('output'),str): raise ValueError('SFT rows need instruction and output strings')
                prefix=tokenizer.encode(template.format(instruction=row['instruction'],input=row.get('input','')))
                response=tokenizer.encode(row['output'])+[tokenizer.eos_id]
                ids=prefix+response
                if len(ids)>seq+1: continue # Do not silently truncate the answer.
                x=ids[:-1]; y=ids[1:]
                if response_only: y=[-100]*max(0,len(prefix)-1)+y[max(0,len(prefix)-1):]
                if not y or all(v==-100 for v in y): continue
                self.rows.append((x+[self.pad]*(seq-len(x)),y+[-100]*(seq-len(y))))
        if not self.rows: raise ValueError('No usable SFT examples; shorten examples or increase sequence length')
    def batch(self,batch_size,generator,device):
        rows=[self.rows[i] for i in torch.randint(len(self.rows),(batch_size,),generator=generator).tolist()]
        return torch.tensor([r[0] for r in rows],device=device),torch.tensor([r[1] for r in rows],device=device)
