import threading,time
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from .generation import generate
from .model import parameter_report

class GenerateRequest(BaseModel):
    prompt: str=Field(max_length=16000)
    max_new_tokens: int=Field(default=100,ge=1,le=512)
    temperature: float=Field(default=0.8,ge=0,le=5)
    top_k: int=Field(default=50,ge=0)
    top_p: float=Field(default=0.95,gt=0,le=1)
    repetition_penalty: float=Field(default=1.0,gt=0,le=10)
class TextRequest(BaseModel):
    text: str=Field(max_length=16000)
class DecodeRequest(BaseModel):
    ids: list[int]=Field(max_length=16000)

def create_app(model,tokenizer):
    app=FastAPI(title='PRIYAN local language model'); lock=threading.Lock()
    @app.get('/health')
    def health(): return {'status':'ok','model_loaded':True}
    @app.get('/model-info')
    def info(): return {'parameters':parameter_report(model),'context_length':model.config.context_length,'tokenizer_vocab_size':tokenizer.vocab_size,'training_status':'User-supplied checkpoint; quality not certified'}
    @app.post('/tokenize')
    def tokenize(r:TextRequest): return {'ids':tokenizer.encode(r.text)}
    @app.post('/decode')
    def decode(r:DecodeRequest):
        if any(i<0 or i>=tokenizer.vocab_size for i in r.ids): raise HTTPException(422,'Token id outside vocabulary')
        return {'text':tokenizer.decode(r.ids)}
    @app.post('/generate')
    def complete(r:GenerateRequest):
        if not lock.acquire(blocking=False): raise HTTPException(429,'Generation busy; retry later')
        try:
            start=time.perf_counter(); result=generate(model,tokenizer,**r.model_dump())
            return {**result,'latency_ms':(time.perf_counter()-start)*1000}
        finally: lock.release()
    return app
