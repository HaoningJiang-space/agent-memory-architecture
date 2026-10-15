"""Fetch pinned official checkpoints through the user-selected HF mirror."""
import json
import os
from pathlib import Path
import time

ROOT = Path('/data/ziheng/agent-memory-architecture')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = str(ROOT / 'hf')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
from huggingface_hub import snapshot_download

MODELS = {
    'Qwen/Qwen3-Embedding-0.6B': '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3',
    'Qwen/Qwen3-8B': 'b968826d9c46dd6066d109eabc6255188de91218',
    'Qwen/Qwen3-32B': '9216db5781bf21249d130ec9da846c4624c16137',
}
manifest_path = ROOT / 'models.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
for name, revision in MODELS.items():
    started = time.time()
    path = snapshot_download(name, revision=revision, max_workers=4,
                             endpoint=os.environ['HF_ENDPOINT'])
    manifest[name] = {'revision': revision, 'path': path,
                      'endpoint': os.environ['HF_ENDPOINT'], 'download_seconds': time.time() - started}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({name: manifest[name]}), flush=True)
