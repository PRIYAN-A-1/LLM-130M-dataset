import os
from llm.generation import load_model
from llm.api import create_app
model, tokenizer, _ = load_model(os.environ["LLM_CHECKPOINT"])
app = create_app(model, tokenizer)
