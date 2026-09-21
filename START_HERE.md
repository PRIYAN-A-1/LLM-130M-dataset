# Start here — own model update

Read OWN_MODEL_GUIDE.md for the new dataset and training workflow.

After installing the project dependencies, run:

```powershell
.\.venv\Scripts\python.exe scripts/chat_own.py --checkpoint starter-demo/latest.pt
```

This runs a **tiny synthetic test checkpoint**, not a useful pretrained 130M chatbot. For your own full model, follow the 130M trial and larger-corpus instructions in OWN_MODEL_GUIDE.md. There is no external pretrained model or API fallback.
