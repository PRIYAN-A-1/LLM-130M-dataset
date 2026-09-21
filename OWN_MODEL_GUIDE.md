# Own 130M model: data → training → replies

This update uses only our own PyTorch Transformer. There is no Ollama, hosted LLM API, downloaded pretrained backbone, or silent fallback. It adds a reproducible starter dataset and a pipeline for language pretraining followed by response-only instruction fine-tuning.

**72 synthetic examples are a pipeline starter, not enough to build a useful conversational model from scratch.** The included new checkpoint is a tiny test model, not a trained 130M assistant. Expect poor or repetitive replies. More training steps on these same few examples do not replace a large, diverse, carefully reviewed corpus.

## New files

- `data/starter/records.jsonl`: 24 concept groups × English, Tamil and Thenglish.
- `data/ready-v1/`: cleaned records, separate SFT/corpus splits, BPE tokenizer, token binaries, provenance and split report.
- `scripts/build_dataset.py`: normalization, exact duplicate removal, group-aware split and training-only tokenizer fitting.
- `scripts/train_own.py`: own-model pretraining followed by SFT.
- `scripts/chat_own.py`: terminal messages and model-generated replies using your checkpoint.
- `scripts/evaluate_own.py`: held-out response loss, perplexity, references and generated samples.
- `starter-demo/`: tiny checkpoint from the actual starter-data test run.

## Windows setup and first run

Extract the ZIP, open PowerShell in `llm-130m`, and install into the Python environment described in README.md. Install a suitable CUDA PyTorch wheel before project dependencies if using your GPU.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/chat_own.py --checkpoint starter-demo/latest.pt
```

Type a question and press Enter; `/quit` exits. This starter chat treats each message independently. It does not claim conversation memory or strong language capability. The model learns the provided Instruction/Input/Response format during SFT; chat uses the same format. Use `start_demo.py` only for the earlier API demo, not this new starter-data checkpoint.

## Run the complete tiny training trial

```powershell
python scripts/train_own.py --stage tiny --data data/ready-v1 --output own-runs/my-tiny --steps 20 --sft-steps 10
python scripts/chat_own.py --checkpoint own-runs/my-tiny/sft/latest.pt
python scripts/evaluate_own.py --checkpoint own-runs/my-tiny/sft/latest.pt --test data/ready-v1/sft/test.jsonl --output my-test-report.json
```

Replace `python` with your virtual environment's Python as needed. Each run needs a fresh output folder. Twenty plus ten steps only validate execution. The pipeline prints actual training and validation metrics, not a readiness certificate.

## Start a short 130M hardware trial

```powershell
python scripts/train_own.py --stage 130m --data data/ready-v1 --output own-runs/my-130m-trial --steps 2 --sft-steps 2
```

This builds 130,046,784 parameters with the original 50,257-output architecture. The starter tokenizer actually contains 1,347 tokens; unused vocabulary slots remain in the model and are masked during generation. This is intentional for the smoke test, not an efficient final vocabulary choice. A serious 130M run should first train its final tokenizer on a sufficiently large training corpus, then initialize a fresh model with that tokenizer. Do not retrain a tokenizer mid-run or reuse incompatible checkpoints.

The trial uses sequence length 256, batch 1, accumulation 1, activation checkpointing and CUDA automatic precision when available. Hardware fit is not guaranteed. If memory fails, adjust a fresh run's config or use a smaller test. The original `python -m llm train` entrypoint supports explicit configs and exact resume. Generated pretrain.yaml/sft.yaml files stay beside every pipeline run. Full-scale training needs a deliberately chosen data/token/compute budget, not just larger step numbers on this starter set.

## Add your own data

Provide UTF-8 JSONL, one record per line:

```json
{"group_id":"variables-001","language":"en","source":"My original course notes","license":"User-owned, authorized for training","text":"A variable refers to a value...","instruction":"Explain a variable.","input":"","output":"A variable is a name that refers to a value..."}
```

Use the **same group_id for translations, paraphrases and related Q&A** so they cannot cross evaluation splits. Include accurate provenance and license information; the tool does not verify those declarations or remove private information automatically. Review all records before training.

```powershell
python scripts/build_dataset.py --input my-data --output data/ready-v2 --vocab-size 8192
```

Use `--vocab-size 50257` for a large corpus intended for the current full architecture. BPE can produce fewer tokens if the corpus has too few distinct merges. Only the train split is used to fit the tokenizer. The three token files are never recombined. Test examples are not used by `train_own.py`; evaluate them after your chosen training plan, not repeatedly as tuning feedback.

The cleaner applies Unicode NFC, normalizes line endings and trims trailing whitespace without destroying code indentation. It removes identical content records. Identical document text or instruction/input pairs merge their groups before splitting. Near-duplicate detection is not implemented; grouping and manual review remain important. Conflicting answers to the same prompt are not automatically resolved—review them.

Splitting sorts seeded group hashes and assigns approximately 80/10/10 by group count. Adding/removing groups may change boundaries: treat a rebuilt dataset as a new version and never resume an older run on it. Output folders cannot be overwritten accidentally. The builder retains records in RAM, so it is for manageable curated corpora; use the original streaming text pipeline or extend preprocessing for very large corpora.

The starter split is 60 train / 6 validation / 6 test records. Every split contains all three languages. Its 2,969 training tokens are **far below** what useful from-scratch language pretraining needs.

## Data license and quality

Starter examples are original AI-generated educational text released with a CC0-1.0 dedication to the extent rights are held. They are labeled synthetic and have not been independently reviewed by a Tamil language expert. No personal documents, private conversation history or external dataset were incorporated. Code retains its MIT license. Your added corpus keeps its own rights and restrictions.

## Evidence and remaining work

See OWN_DATA_VALIDATION.md and own-test-report.json. The tiny pipeline is exercised with real loss calculations and saved weights. Its responses are not certified as useful. Full 130M training on a substantive corpus, multilingual quality evaluation, Windows/CUDA execution and 6 GB GPU fit remain to be done. The older VALIDATION.md describes earlier implementation tests separately.
