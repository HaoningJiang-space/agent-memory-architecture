"""Run a fresh native 32B Mem0 control after the isolated servers are ready."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
out = root/'outputs/phase2'
out.mkdir(parents=True, exist_ok=True)
receipt = out/'native_control_pipeline.json'
if receipt.exists():
    raise RuntimeError('Refusing to overwrite an existing phase-2 run')
command = [sys.executable, '-u', str(root/'run_native.py'), '--algorithm', 'mem0',
           '--run', 'phase2/mem0_32b_native', '--stage', 'both',
           '--construction-model', 'Qwen/Qwen3-32B', '--answer-max-tokens', '1024',
           '--trace-file', 'outputs/phase2/api_trace.jsonl']
record = dict(start=time.time(), status='waiting_for_services', command=command,
              scope='One complete real history; native algorithm; prefix caching enabled. '
                    'Completion and reference validity do not establish semantic quality.')
receipt.write_text(json.dumps(record, indent=2)+'\n')
try:
    deadline = time.monotonic()+900
    for port in (8030, 8032, 8033):
        while True:
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).read()
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Service {port} readiness timeout')
                time.sleep(5)
    record.update(status='running', ready=time.time())
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    result = subprocess.run(command, cwd=root, check=False)
    record.update(status='complete' if result.returncode == 0 else 'failed',
                  returncode=result.returncode)
except Exception as exc:
    record.update(status='failed', error=repr(exc))
    raise
finally:
    record['end'] = time.time()
    receipt.write_text(json.dumps(record, indent=2)+'\n')
