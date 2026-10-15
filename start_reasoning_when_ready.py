"""Start only after the pinned 32B download has committed its manifest entry."""
import json
from pathlib import Path
import subprocess
import sys
import time

r=Path(__file__).resolve().parent
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
    if 'Qwen/Qwen3-32B' in json.loads((r/'models.json').read_text()):
        subprocess.run([sys.executable,str(r/'launch_services.py'),'reasoning'],check=True,cwd=r)
        break
    time.sleep(20)
else:
    raise RuntimeError('Model download readiness deadline exceeded')
