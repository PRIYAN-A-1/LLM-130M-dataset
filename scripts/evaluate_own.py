import argparse,json,math
from pathlib import Path
import torch
from llm.generation import load_model,generate
from llm.data import InstructionData,DEFAULT_TEMPLATE

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--test',default='data/ready-v1/sft/test.jsonl');p.add_argument('--output',default='own-evaluation.json');a=p.parse_args()
model,tok,c=load_model(a.checkpoint);device=next(model.parameters()).device
model.config.context_length=min(model.config.context_length,c.training.sequence_length)
d=InstructionData(a.test,tok,c.training.sequence_length)
summed=0.;count=0
with torch.inference_mode():
 for x,y in d.rows:
  x=torch.tensor([x],device=device);y=torch.tensor([y],device=device)
  _,loss=model(x,y);n=(y!=-100).sum().item();summed+=loss.item()*n;count+=n
records=[json.loads(s) for s in Path(a.test).read_text(encoding='utf-8').splitlines() if s.strip()]
samples=[]
for r in records:
 prompt=DEFAULT_TEMPLATE.format(instruction=r['instruction'],input=r.get('input',''))
 if len(tok.encode(prompt))>=model.config.context_length:continue
 samples.append({'language':r['language'],'question':r['instruction'],'reference':r['output'],'generated':generate(model,tok,prompt,max_new_tokens=32,temperature=0)['completion']})
result={'test_loss':summed/count,'test_perplexity':math.exp(summed/count),'examples_evaluated':len(d.rows),'examples_total':len(records),'skipped_too_long':len(records)-len(d.rows),'scope':'Small held-out starter set; not a broad quality benchmark','samples':samples}
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
