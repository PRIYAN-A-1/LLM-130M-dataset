"""Clean a local corpus, group-split it, and train BPE on training text only."""
import argparse,collections,hashlib,json,tempfile,unicodedata
from pathlib import Path
import numpy as np
from llm.tokenizer import BPETokenizer

def normalize(s):
    if not isinstance(s,str): raise ValueError('Text fields must be strings')
    return '\n'.join(x.rstrip() for x in unicodedata.normalize('NFC',s).replace('\r\n','\n').replace('\r','\n').split('\n')).strip()

def build(source,out,vocab_size=8192,seed=42):
    source=Path(source); out=Path(out)
    if out.exists(): raise ValueError('Output exists; choose a new version directory')
    files=sorted(source.rglob('*.jsonl')) if source.is_dir() else [source]
    rows=[]; seen={}; parent={}; duplicates=0
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b): parent[find(b)]=find(a)
    for path in files:
        with path.open(encoding='utf-8') as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                row=json.loads(line)
                for key in ('group_id','language','source','license','text','instruction','output'):
                    if not isinstance(row.get(key),str) or not row[key].strip(): raise ValueError(f'{path}:{line_no}: missing {key}')
                row={**row,'text':normalize(row['text']),'instruction':normalize(row['instruction']),'input':normalize(row.get('input','')),'output':normalize(row['output'])}
                if not all(row[k] for k in ('text','instruction','output')): raise ValueError('Empty normalized record')
                find(row['group_id'])
                # Group identical prompts or documents, including their translations/siblings.
                for key in ('text','prompt'):
                    value=row['text'] if key=='text' else row['instruction']+'\0'+row['input']
                    digest=(key,hashlib.sha256(value.encode()).hexdigest())
                    if digest in seen: union(row['group_id'],seen[digest])
                    else: seen[digest]=row['group_id']
                rows.append(row)
    unique=[]; seen_rows=set()
    for row in rows:
        digest=hashlib.sha256(json.dumps({k:row[k] for k in ('text','instruction','input','output')},ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if digest in seen_rows: duplicates+=1; continue
        seen_rows.add(digest); unique.append(row)
    groups=sorted({find(r['group_id']) for r in unique},key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    if len(groups)<10: raise ValueError('Need at least 10 distinct concept groups for useful three-way splitting')
    n=max(1,len(groups)//10)
    assignment={g:('test' if i<n else 'validation' if i<2*n else 'train') for i,g in enumerate(groups)}
    splits={k:[] for k in ('train','validation','test')}
    for row in unique:
        row['split_group']=find(row['group_id']); splits[assignment[row['split_group']]].append(row)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage=Path(temp)/'dataset'; stage.mkdir()
        for folder in ('corpus','sft','tokens'): (stage/folder).mkdir()
        for split,records in splits.items():
            for folder in ('corpus','sft'):
                with (stage/folder/f'{split}.jsonl').open('w',encoding='utf-8') as f:
                    for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        tok=BPETokenizer.train((r['text'] for r in splits['train']),stage/'tokenizer.json',vocab_size)
        counts={}; hashes={}
        for split,records in splits.items():
            digest=hashlib.sha256(); count=0
            with (stage/'tokens'/f'{split}.bin').open('wb') as f:
                for row in records:
                    ids=tok.encode(row['text'])+[tok.eos_id]; raw=np.asarray(ids,dtype='<u4').tobytes()
                    f.write(raw); digest.update(raw); count+=len(ids)
            counts[split]=count;hashes[split]=digest.hexdigest()
        meta={'counts':counts,'sha256':hashes,'tokenizer_hash':tok.fingerprint,'vocab_size':tok.vocab_size,'dtype':'uint32-little-endian','seed':seed}
        (stage/'tokens/metadata.json').write_text(json.dumps(meta,indent=2))
        report={'input_records':len(rows),'duplicates_removed':duplicates,'records':{k:len(v) for k,v in splits.items()},'groups':{k:len({r['split_group'] for r in v}) for k,v in splits.items()},'languages':{k:dict(collections.Counter(r['language'] for r in v)) for k,v in splits.items()},'tokens':counts,'tokenizer_vocab':tok.vocab_size,'tokenizer_training_split':'train only','scope':'Starter data; no claim of chat quality','source_files':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
        (stage/'report.json').write_text(json.dumps(report,indent=2))
        stage.rename(out)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/starter/records.jsonl');p.add_argument('--output',default='data/ready-v1');p.add_argument('--vocab-size',type=int,default=8192);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    print(json.dumps(build(a.input,a.output,a.vocab_size,a.seed),indent=2))
