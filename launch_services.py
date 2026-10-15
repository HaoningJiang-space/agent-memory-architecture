"""Start requested local calibration services, recording exact argv and PIDs."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
models = json.loads((ROOT/'models.json').read_text())
python = str(ROOT/'.venv/bin/python')
record = ROOT/'services.json'
manifest = json.loads(record.read_text()) if record.exists() else {}
for name in sys.argv[1:]:
    assert name in ['proxy', 'embedding', 'construction', 'reasoning']
    if name in manifest:
        try:
            os.kill(manifest[name]['pid'], 0)
            print(name, 'already running')
            continue
        except ProcessLookupError:
            pass
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE='1', HF_HOME=str(ROOT/'hf'), TMPDIR=str(ROOT/'tmp'),
               VLLM_CACHE_ROOT=str(ROOT/'vllm-cache'), TORCHINDUCTOR_CACHE_DIR=str(ROOT/'torch-cache'),
               CUDA_CACHE_PATH=str(ROOT/'cuda-cache'), VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
               CUDA_HOME='/usr/local/cuda-12.8', VLLM_USE_FLASHINFER_SAMPLER='0',
               VLLM_ATTENTION_BACKEND='FLASH_ATTN', XDG_CACHE_HOME=str(ROOT/'cache'))
    env['PATH'] = '/usr/local/cuda-12.8/bin:' + env['PATH']
    if name == 'proxy':
        args = [python, '-m', 'uvicorn', 'trace_proxy:app', '--host', '127.0.0.1', '--port', '8030']
    elif name == 'embedding':
        env['CUDA_VISIBLE_DEVICES'] = '0'
        args = [python, '-m', 'uvicorn', 'embedding_server:app', '--host', '127.0.0.1', '--port', '8033']
    else:
        m = 'Qwen/Qwen3-8B' if name == 'construction' else 'Qwen/Qwen3-32B'
        env['CUDA_VISIBLE_DEVICES'] = '0' if name == 'construction' else '1'
        args = [python, '-m', 'vllm.entrypoints.openai.api_server', '--model', models[m]['path'],
                '--served-model-name', m, '--host', '127.0.0.1', '--port', '8031' if name == 'construction' else '8032',
                '--dtype', 'bfloat16', '--max-model-len', '32768', '--max-num-seqs', '16',
                '--max-num-batched-tokens', '8192', '--gpu-memory-utilization', '0.60' if name == 'construction' else '0.95',
                '--generation-config', 'vllm', '--no-enable-prefix-caching', '--disable-log-requests']
    log = (ROOT/'logs'/f'{name}.log').open('ab')
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    manifest[name] = {'pid':p.pid, 'argv':args,
                      'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','HF_HOME','TMPDIR','VLLM_CACHE_ROOT','CUDA_HOME','VLLM_USE_FLASHINFER_SAMPLER','VLLM_ATTENTION_BACKEND'] if k in env}}
    record.write_text(json.dumps(manifest, indent=2)+'\n')
    print(name, p.pid)
