"""One pipeline, only the project's own randomly initialized Transformer."""
import argparse,json,subprocess,sys
from pathlib import Path
import yaml
from llm.config import Config

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['tiny','130m'],default='tiny');p.add_argument('--data',default='data/ready-v1');p.add_argument('--output',default='own-runs/tiny-v1');p.add_argument('--steps',type=int,default=20);p.add_argument('--sft-steps',type=int,default=10);a=p.parse_args()
    if min(a.steps,a.sft_steps)<2:p.error('Use at least 2 steps for each stage')
    data=Path(a.data).resolve();out=Path(a.output).resolve()
    if out.exists():p.error('Choose a fresh output directory; existing runs are preserved')
    report=json.loads((data/'report.json').read_text())
    config=Config.load('configs/laptop.yaml').dict()
    if a.stage=='tiny':
        config['model'].update(vocab_size=max(512,report['tokenizer_vocab']),embedding_dim=128,num_layers=2,num_heads=4,mlp_dim=512,context_length=256)
    if report['tokenizer_vocab']>config['model']['vocab_size']:p.error('Tokenizer is too large for model')
    config['training'].update(batch_size=1,sequence_length=256,gradient_accumulation_steps=1,max_steps=a.steps,warmup_steps=min(2,a.steps-1),eval_batches=2,eval_interval=max(1,a.steps//2),checkpoint_interval=max(1,a.steps//2),log_interval=1)
    if min(report['tokens'].values())<=256:p.error('Each split needs more than 256 tokens; add distinct data')
    out.mkdir(parents=True)
    pre=out/'pretrain.yaml';pre.write_text(yaml.safe_dump(config))
    def run(*args):subprocess.run([sys.executable,'-m','llm',*map(str,args)],check=True)
    run('train','--config',pre,'--tokenizer',data/'tokenizer.json','--data',data/'tokens','--output',out/'pretrain')
    config['training'].update(max_steps=a.sft_steps,warmup_steps=min(2,a.sft_steps-1),learning_rate=3e-5,min_lr=3e-6)
    fine=out/'sft.yaml';fine.write_text(yaml.safe_dump(config))
    run('finetune','--config',fine,'--tokenizer',data/'tokenizer.json','--data',data/'sft','--instructions',data/'sft/train.jsonl','--checkpoint',out/'pretrain/latest.pt','--output',out/'sft')
    print('Training completed. This is a short experiment, not a chat-quality certification.')
    print('Checkpoint:',out/'sft/latest.pt')
    print('Use scripts/chat_own.py with that checkpoint; evaluate using scripts/evaluate_own.py.')
if __name__=='__main__':main()
