from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3376
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    attention_backend: str = 'sdpa'
    def __post_init__(self):
        for k in ('vocab_size','context_length','embedding_dim','num_layers','num_heads','mlp_dim'):
            if getattr(self,k) <= 0: raise ValueError(f'{k} must be positive')
        if self.embedding_dim % self.num_heads: raise ValueError('embedding_dim must be divisible by num_heads')
        if not 0 <= self.dropout < 1: raise ValueError('dropout must be in [0,1)')
        if self.attention_backend not in ('sdpa','manual'): raise ValueError('invalid attention_backend')

@dataclass
class TrainConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    sequence_length: int = 256
    learning_rate: float = 0.0003
    min_lr: float = 0.00003
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 100
    log_interval: int = 10
    precision: str = 'auto'
    seed: int = 42
    deterministic: bool = False
    def __post_init__(self):
        for k in ('batch_size','gradient_accumulation_steps','sequence_length','max_steps','eval_interval','eval_batches','checkpoint_interval','log_interval'):
            if getattr(self,k) < 1: raise ValueError(f'{k} must be positive')
        if not 0 <= self.warmup_steps < self.max_steps: raise ValueError('warmup_steps must be below max_steps')
        if not 0 <= self.min_lr <= self.learning_rate or self.learning_rate <= 0: raise ValueError('invalid learning rates')
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or self.weight_decay < 0: raise ValueError('invalid optimizer settings')
        if self.precision not in ('auto','fp32','fp16','bf16'): raise ValueError('invalid precision')

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    def __post_init__(self):
        if self.training.sequence_length > self.model.context_length: raise ValueError('sequence_length exceeds model context')
    @classmethod
    def from_dict(cls,d): return cls(ModelConfig(**d.get('model',{})),TrainConfig(**d.get('training',{})))
    @classmethod
    def load(cls,path):
        with open(path,encoding='utf-8') as f: return cls.from_dict(yaml.safe_load(f))
    def dict(self): return asdict(self)
