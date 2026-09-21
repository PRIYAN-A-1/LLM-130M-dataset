# Executed validation — 20 September 2026

## Environment
Python 3.12; PyTorch 2.14.0+cu130; CPU execution; no accessible CUDA GPU. See requirements-tested.txt for direct package versions. CPU tests do not prove Windows/CUDA compatibility or 6 GB GPU memory fit.

## Results

| Check | Actual result |
|---|---|
| Unit/integration suite | 10 passed; two dependency deprecation warnings |
| UTF-8 BPE round-trip | English, Tamil and emoji passed |
| Causal attention | Changing future tokens preserves earlier outputs |
| Explicit attention vs SDPA | Outputs agree within test tolerance |
| Tiny fixed-batch overfit | Loss 5.733703 to 0.017348 after 60 updates |
| Gradients and activation checkpointing | Nonzero gradients; backward passed |
| Exact CPU resume | Interrupted 2+2 steps equals uninterrupted 4 steps, tensor-for-tensor |
| API TestClient | Health, model info, tokenize, decode, generate and invalid-input rejection passed |
| Samplers | Greedy path, top-k=1 and restrictive top-p tested |
| Data windows | Next-token alignment tested |
| Fine-tuning masks | Response-only mask and EOS targets tested |
| Parameter counter | 130,046,784 unique trainable parameters |
| Full-size CPU forward | Tensor output shape (1, 4, 50257), finite loss |
| Tiny CLI pretraining | 50 optimizer steps completed |
| CLI evaluation | Validation loss 3.452505588531494; perplexity 31.57941829240989 |
| CLI generation | Completed 20 new tokens; text is not coherent general language |
| CLI safetensors export | Completed; reloaded weights match checkpoint exactly |
| Tiny SFT CLI | 2 updates completed; validation loss 5.778999328613281 (toy test only) |

The overfit test uses a much smaller fixed-batch model than the 50-step CLI demo. These losses are from different tests and must not be combined into one training curve. The tiny CLI corpus and tokenizer are synthetic demonstration artifacts, not a research benchmark. No hardware-independent throughput claims are made.

## Not verified or not trained

- Full 130M pretraining and instruction quality evaluation have not been run. Full-model backward was subsequently verified in a two-step CPU smoke test.
- CUDA FP16/BF16, CUDA checkpoint resume, MPS and Windows execution are not tested here.
- A two-process CPU DDP smoke test was attempted and failed during Gloo process-group initialization: “Operation not permitted” in this environment. No distributed training step ran. DDP implementation is unverified; this is not a passing test.
- No multi-GPU run, GPU memory benchmark, broad evaluation benchmark or production API load test was performed.
- Notebooks are educational source examples, not claimed as executed notebooks.

## Reproduce

From the project root after installation:

```bash
python scripts/count_parameters.py
python -m pytest -q -s
python -m llm generate --checkpoint demo/latest.pt --prompt "Artificial intelligence" --max-new-tokens 20
```

Run the README pipeline with a fresh output directory to reproduce full CLI checks. Generation samples are stochastic. The package includes no 130M weights and no claim of a ready conversational assistant.

## Additional full-model execution test

130,046,784 parameters: two CPU FP32 optimizer steps completed with batch 1, accumulation 1, 32-token windows, activation checkpointing and a 512-token synthetic-data tokenizer within the 50,257-output model. Training losses were 10.929885864257812 and 8.495359420776367. The saved step-2 checkpoint reloaded with exact tensor equality. End-to-end measured time was 33.45 seconds in this environment, not a laptop speed prediction. See FULL_MODEL_SMOKE.json. This does not establish useful language ability, convergence, full-context training, or GPU memory fit. The large synthetic smoke checkpoint is not included in the ZIP.
