# Executed starter-data validation

- 72 original synthetic examples; 24 grouped topics; English, Tamil and Thenglish.
- 60 training, 6 validation and 6 test examples; group disjointness verified.
- Training-only BPE tokenizer: 1,347 tokens.
- 12 tests passed, including original model tests plus dataset deduplication and split-integrity checks. Two dependency deprecation warnings.
- Tiny own model: 20 pretraining steps, then 10 response-only fine-tuning steps on CPU.
- Held-out test: 6 of 6 examples evaluated; 0 skipped.
- Test response loss: 6.829219066544084; test perplexity: 924.4685820641394.
- Terminal chat executed with a Thenglish input and generated an output. This is execution verification, not language-quality success.
- Generated samples and references are in own-test-report.json. Output remains poor after this short trial.
- No full 130M run on the new corpus, CUDA/Windows test, or GPU-fit verification was performed in this update.

The included starter-demo checkpoint is the tiny model from this run. The pre-existing demo checkpoint belongs to an older test. Neither is a useful pretrained 130M assistant.
