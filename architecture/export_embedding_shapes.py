"""Run on moe-server: export only shape metadata from frozen native traces."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    trace = (args.root/'outputs/api_trace.jsonl').read_bytes()
    receipt = json.loads((args.root/'outputs/raw_artifact_sha256.json').read_text())
    assert hashlib.sha256(trace).hexdigest()==receipt['files']['outputs/api_trace.jsonl']['sha256']
    raw = (args.root/'outputs/embedding_kernels.jsonl').read_bytes()
    kernels = [json.loads(line) for line in raw.splitlines()]
    calls = []
    for r in map(json.loads,trace.splitlines()):
        c = r.get('context',{})
        selected = ((c.get('run')=='mem0_full_01' and c.get('phase')=='construction' and c.get('index',999)<9)
                    or c.get('run')=='embed_full_01')
        if r.get('path')!='embeddings' or not selected: continue
        batches = [k for k in kernels if r['start']<=k['start'] and k['end']<=r['end']]
        assert sum(k['batch'] for k in batches)==r['embedding_count']
        calls.append(dict(trace_id=r['id'],context=c,
                          batches=[{key:k[key] for key in ('batch','lengths','padded_length')} for k in batches]))
    output = dict(scope='Real native embedding kernel shapes; no text, vectors or responses exported.',
                  trace_sha256=hashlib.sha256(trace).hexdigest(),kernel_sha256=hashlib.sha256(raw).hexdigest(),calls=calls)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
