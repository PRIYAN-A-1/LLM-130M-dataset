"""Terminal chat using ONLY a custom trained project checkpoint."""
import argparse
from llm.generation import load_model,generate
from llm.data import DEFAULT_TEMPLATE

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--max-new-tokens',type=int,default=64);a=p.parse_args()
model,tok,c=load_model(a.checkpoint)
# Avoid relying on positional embeddings beyond the trained window.
model.config.context_length=min(model.config.context_length,c.training.sequence_length)
print('Own model terminal chat. Short training can produce nonsense. /quit exits.')
print('Each message is independent; multi-turn memory is not trained in this starter.')
while True:
    try:text=input('You: ').strip()
    except (EOFError,KeyboardInterrupt):break
    if text=='/quit':break
    if not text:continue
    prompt=DEFAULT_TEMPLATE.format(instruction=text,input='')
    if len(tok.encode(prompt))>=model.config.context_length:
        print('Please shorten your message to fit the trained context.');continue
    print('PRIYAN:',generate(model,tok,prompt,max_new_tokens=a.max_new_tokens)['completion'])
