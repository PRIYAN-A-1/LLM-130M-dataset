from pathlib import Path
import hashlib
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
SPECIAL=['<BOS>','<EOS>','<PAD>','<UNK>']
class BPETokenizer:
    def __init__(self,tokenizer): self.backend=tokenizer
    @classmethod
    def train(cls,documents,path,vocab_size=50257):
        if vocab_size<260: raise ValueError('Byte-level BPE requires vocab_size >=260')
        t=Tokenizer(models.BPE(unk_token='<UNK>'))
        t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        t.decoder=decoders.ByteLevel()
        t.train_from_iterator(documents,trainers.BpeTrainer(vocab_size=vocab_size,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
        obj=cls(t); obj.save(path); return obj
    @classmethod
    def load(cls,path):
        obj=cls(Tokenizer.from_file(str(path)))
        if any(obj.backend.token_to_id(s) is None for s in SPECIAL): raise ValueError('Missing special tokens')
        return obj
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True); self.backend.save(str(path))
    def encode(self,text): return self.backend.encode(text,add_special_tokens=False).ids
    def decode(self,ids): return self.backend.decode(ids,skip_special_tokens=True)
    def batch_encode(self,texts): return [x.ids for x in self.backend.encode_batch(texts,add_special_tokens=False)]
    @property
    def eos_id(self): return self.backend.token_to_id('<EOS>')
    @property
    def bos_id(self): return self.backend.token_to_id('<BOS>')
    @property
    def vocab_size(self): return self.backend.get_vocab_size()
    @property
    def fingerprint(self): return hashlib.sha256(self.backend.to_str().encode()).hexdigest()
