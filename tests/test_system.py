from dataclasses import replace
import json
import pytest
import torch
from fastapi.testclient import TestClient
from llm.config import Config,ModelConfig,TrainConfig
from llm.model import LanguageModel,parameter_report
from llm.tokenizer import BPETokenizer
from llm.data import prepare,TokenData,InstructionData
from llm.generation import generate,sample,load_model
from llm.training import train,load_checkpoint,lr_at
from llm.api import create_app

torch.set_num_threads(1)

def cfg():
    return Config(ModelConfig(vocab_size=300,context_length=32,embedding_dim=32,num_layers=2,num_heads=4,mlp_dim=64,gradient_checkpointing=False),TrainConfig(batch_size=2,gradient_accumulation_steps=1,sequence_length=16,max_steps=4,warmup_steps=1,eval_interval=2,eval_batches=1,checkpoint_interval=2,log_interval=2,precision='fp32'))

@pytest.fixture
def corpus(tmp_path):
    source=tmp_path/'text.txt'
    source.write_text(''.join(f'Example {i}: the cat reads a book and learns today.\n' for i in range(100)))
    tok=BPETokenizer.train((f'Example {i}: the cat reads a book and learns today.' for i in range(100)),tmp_path/'tok.json',300)
    prepare(source,tok,tmp_path/'data')
    return tok,tmp_path/'data'

def test_tokenizer_roundtrip(corpus):
    tok,_=corpus
    text='Hello Tamil தமிழ்! 😀'
    assert tok.decode(tok.encode(text))==text
    assert tok.batch_encode([text])==[tok.encode(text)]

def test_causal_and_fallback():
    c=cfg().model; model=LanguageModel(c).eval()
    x=torch.randint(300,(2,12)); y=x.clone(); y[:,7:]=torch.randint(300,(2,5))
    assert torch.allclose(model(x)[0][:,:7],model(y)[0][:,:7],atol=1e-6)
    other=LanguageModel(replace(c,attention_backend='manual')).eval(); other.load_state_dict(model.state_dict())
    assert torch.allclose(model(x)[0],other(x)[0],atol=1e-6)

def test_overfit_and_gradients():
    torch.manual_seed(5); model=LanguageModel(cfg().model)
    x=torch.tensor([[4,5,6,7]*4]); y=torch.tensor([[5,6,7,4]*4])
    opt=torch.optim.AdamW(model.parameters(),lr=0.01)
    initial=model(x,y)[1].item()
    for _ in range(60):
        opt.zero_grad(); loss=model(x,y)[1]; loss.backward()
        assert model.token_embedding.weight.grad.abs().sum()>0
        opt.step()
    assert loss.item()<initial*0.15
    print(f'overfit initial={initial:.6f} final={loss.item():.6f}')

def test_count():
    c=Config.load('configs/130m.yaml')
    with torch.device('meta'): m=LanguageModel(c.model)
    r=parameter_report(m)
    expected=c.model.vocab_size*768+1024*768+12*(4*768*768+4*768+2*768*3376+3376+768+4*768)+2*768
    assert r['total']==expected
    assert 129_000_000<r['total']<131_000_000
    assert sum(r[k] for k in ('embedding','attention','mlp','layernorm','lm_head_unique'))==r['total']

def test_generation_and_api(corpus):
    tok,_=corpus; model=LanguageModel(cfg().model).eval()
    result=generate(model,tok,'Hello',max_new_tokens=3,temperature=0,stop_tokens=[])
    assert result['tokens_generated']==3
    assert sample(torch.tensor([[1.,8.,0.]]),top_k=1).item()==1
    assert sample(torch.tensor([[1.,8.,0.]]),top_k=0,top_p=0.001).item()==1
    with pytest.raises(ValueError): sample(torch.zeros(1,3),top_p=0)
    client=TestClient(create_app(model,tok))
    assert client.get('/health').status_code==200
    assert client.get('/model-info').status_code==200
    ids=client.post('/tokenize',json={'text':'Hi'}).json()['ids']
    assert client.post('/decode',json={'ids':ids}).json()['text']=='Hi'
    assert client.post('/decode',json={'ids':[-1]}).status_code==422
    assert client.post('/generate',json={'prompt':'Hi','max_new_tokens':2}).status_code==200
    assert client.post('/generate',json={'prompt':'Hi','top_p':0}).status_code==422

def test_data_targets(corpus):
    _,data=corpus; d=TokenData(data/'train.bin',16)
    x,y=d.batch(2,torch.Generator().manual_seed(1),torch.device('cpu'))
    assert torch.equal(x[:,1:],y[:,:-1])

def test_resume_exact(corpus,tmp_path):
    tok,data=corpus; c=cfg()
    train(c,tok,data,tmp_path/'whole')
    train(c,tok,data,tmp_path/'resumed',stop_after=2)
    train(c,tok,data,tmp_path/'resumed',resume=True)
    a=load_checkpoint(tmp_path/'whole/latest.pt'); b=load_checkpoint(tmp_path/'resumed/latest.pt')
    assert a['step']==b['step']==4
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    m,t,_=load_model(tmp_path/'resumed/latest.pt')
    assert generate(m,t,'Hello',max_new_tokens=2)['tokens_generated']>0

def test_sft_mask(corpus,tmp_path):
    tok,_=corpus; p=tmp_path/'sft.jsonl'
    p.write_text(json.dumps({'instruction':'Hi','input':'','output':'Hello'})+'\n')
    d=InstructionData(p,tok,128)
    x,y=d.batch(1,torch.Generator(),torch.device('cpu'))
    assert (y==-100).any() and (y!=-100).any()
    assert tok.eos_id in y

def test_scheduler_and_config():
    c=cfg(); assert lr_at(3,c.training)==pytest.approx(c.training.min_lr)
    with pytest.raises(ValueError): ModelConfig(embedding_dim=7,num_heads=4)
    with pytest.raises(ValueError): TrainConfig(max_steps=10,warmup_steps=10)

def test_checkpointing_backward():
    model=LanguageModel(replace(cfg().model,gradient_checkpointing=True))
    x=torch.randint(300,(1,16)); model(x,x)[1].backward()
    assert all(p.grad is not None for p in model.parameters())
