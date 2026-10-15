"""Recreate the pinned local MAB input through the requested mirror."""
import hashlib
import json
import os
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
os.environ.update(HF_ENDPOINT='https://hf-mirror.com',HF_HOME=str(root/'hf'),HF_HUB_DISABLE_XET='1')
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
revision='7ea066982b140a19337e17e60d45d4076e042faf'
path=hf_hub_download('ai-hyz/MemoryAgentBench','data/Accurate_Retrieval-00000-of-00001.parquet',
    repo_type='dataset',revision=revision,endpoint='https://hf-mirror.com',token=False)
records=pq.read_table(path).to_pylist()
selected=[x for x in records if x.get('metadata',{}).get('source','').startswith('longmemeval_s')]
assert len(selected)==5,len(selected)
target=root/'longmemeval.json'
if target.exists():
    assert json.loads(target.read_text())==selected,'Existing input differs; refuse overwrite'
else:
    target.write_text(json.dumps(selected,ensure_ascii=False)+'\n')
print(json.dumps(dict(samples=len(selected),revision=revision,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_path=path)))
