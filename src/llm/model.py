import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.qkv=nn.Linear(c.embedding_dim,3*c.embedding_dim)
        self.proj=nn.Linear(c.embedding_dim,c.embedding_dim)
    def forward(self,x):
        b,t,d=x.shape; h=self.c.num_heads
        q,k,v=self.qkv(x).view(b,t,3,h,d//h).permute(2,0,3,1,4).unbind(0)
        drop=self.c.dropout if self.training else 0.0
        if self.c.attention_backend=='sdpa':
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=drop)
        else:
            scores=q @ k.transpose(-2,-1)/math.sqrt(d//h)
            mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
            probs=scores.masked_fill(~mask,float('-inf')).softmax(-1)
            y=F.dropout(probs,p=drop,training=self.training) @ v
        return self.proj(y.transpose(1,2).contiguous().view(b,t,d))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.ln1=nn.LayerNorm(c.embedding_dim); self.ln2=nn.LayerNorm(c.embedding_dim)
        self.attention=Attention(c)
        self.mlp=nn.Sequential(nn.Linear(c.embedding_dim,c.mlp_dim),nn.GELU(),nn.Linear(c.mlp_dim,c.embedding_dim))
        self.drop=nn.Dropout(c.dropout)
    def forward(self,x):
        x=x+self.drop(self.attention(self.ln1(x)))
        return x+self.drop(self.mlp(self.ln2(x)))

class LanguageModel(nn.Module):
    def __init__(self,c):
        super().__init__(); self.config=c
        self.token_embedding=nn.Embedding(c.vocab_size,c.embedding_dim)
        self.position_embedding=nn.Embedding(c.context_length,c.embedding_dim)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.final_norm=nn.LayerNorm(c.embedding_dim); self.drop=nn.Dropout(c.dropout)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.attention.proj.weight,std=0.02/math.sqrt(2*c.num_layers))
            nn.init.normal_(block.mlp[2].weight,std=0.02/math.sqrt(2*c.num_layers))
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.LayerNorm): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self,ids,targets=None):
        if ids.ndim!=2 or not 0<ids.shape[1]<=self.config.context_length: raise ValueError('Expected [batch, tokens] within context_length')
        x=self.drop(self.token_embedding(ids)+self.position_embedding(torch.arange(ids.shape[1],device=ids.device)))
        for block in self.blocks:
            x=checkpoint(block,x,use_reentrant=False) if self.training and self.config.gradient_checkpointing else block(x)
        logits=F.linear(self.final_norm(x),self.token_embedding.weight)
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,self.config.vocab_size),targets.reshape(-1),ignore_index=-100)
        return logits,loss

def parameter_report(model):
    result=dict(embedding=0,attention=0,mlp=0,layernorm=0,lm_head_unique=0)
    for name,p in model.named_parameters():
        key='embedding' if 'embedding' in name else 'attention' if 'attention' in name else 'mlp' if 'mlp' in name else 'layernorm'
        result[key]+=p.numel()
    result['total']=sum(p.numel() for p in model.parameters())
    result['trainable']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return result
