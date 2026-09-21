"""Start the included tiny demonstration API after installing requirements."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
checkpoint = root / 'demo' / 'latest.pt'
if not checkpoint.is_file():
    raise SystemExit('Missing demo checkpoint. Extract the entire project ZIP first.')
try:
    import llm
    import uvicorn
except ImportError:
    raise SystemExit('Install the project first: python -m pip install -r requirements.txt')
print('Starting the TINY synthetic demo, not a trained 130M assistant.', flush=True)
print('Open http://127.0.0.1:8000/docs after server startup. Ctrl+C stops it.', flush=True)
try:
    subprocess.run([sys.executable, '-m', 'llm', 'serve', '--checkpoint', str(checkpoint)], cwd=root, check=True)
except KeyboardInterrupt:
    pass
