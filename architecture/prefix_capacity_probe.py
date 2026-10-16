"""Exact-prefix capacity accounting on existing private prompts; no GPU run.

Export counts/hashes only. Potential reusable tokens are not measured cache hits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def prefix_slots(sequences, block_size=1):
    edges, partial = {}, 0
    for sequence in sequences:
        parent = 0
        for begin in range(0, len(sequence), block_size):
            block = tuple(sequence[begin:begin+block_size])
            if len(block) < block_size:
                partial += 1
                break
            key = (parent, block)
            if key not in edges:
                edges[key] = len(edges)+1
            parent = edges[key]
    return (len(edges)+partial)*block_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--source-root', default='/data/ziheng/agent-memory-architecture')
    args = p.parse_args()
    output = Path(args.output)
    assert not output.exists()
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    from transformers import AutoTokenizer
    root = Path(args.source_root)
    model = json.loads((root / 'models.json').read_text())['Qwen/Qwen3-32B']
    tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True)
    groups = {s: [] for s in ('raw', 'raw_512', 'mem0')}
    sources = [('finite_prefix', 'finite_prefix_01'), ('representation_controls', 'representation_01')]
    for name, run in sources:
        frozen = json.loads((ROOT / f'evidence/architecture/{name}.json').read_text())
        for row in frozen['summary']['results']:
            index, strategy = row['index'], row['strategy']
            name = f'{index:02d}_{strategy}_private.json' if run == 'finite_prefix_01' else f'{index:02d}_private.json'
            raw = (root / 'outputs' / run / name).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == (row.get('private_receipt_sha256') or row.get('private_sha256'))
            receipt = json.loads(raw)
            tokens = tokenizer.apply_chat_template(receipt['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            assert len(tokens) == row['input_tokens']
            groups[strategy].append((index, tokens, digest))
    # Independent hand counts: shared prefix [1,2], two private suffixes.
    assert prefix_slots([[1, 2, 3], [1, 2, 4]]) == 4
    assert prefix_slots([[1, 2, 3], [1, 2, 4]], 2) == 6
    # Equal suffixes after different prefixes are not shareable KV.
    assert prefix_slots([[1, 3], [2, 3]]) == 4
    rows = []
    for strategy, calls in groups.items():
        calls.sort()
        assert [c[0] for c in calls] == [3, 28, 30, 31, 39, 48, 54]
        for batch in (2, 4, 7):
            sequences = [c[1] for c in calls[:batch]]
            private = sum(map(len, sequences))
            exact = prefix_slots(sequences)
            assert exact == prefix_slots(list(reversed(sequences))) <= private
            rows.append(dict(strategy=strategy, batch=batch, indices=[c[0] for c in calls[:batch]],
                source_sha256=[c[2] for c in calls[:batch]],
                private_prompt_tokens=private, unique_prefix_tokens=exact,
                ideal_prompt_capacity_saved_fraction=1-exact/private,
                block_controls=[dict(block_size=size, shared_slots=prefix_slots(sequences, size),
                    unshared_slots=sum((len(s)+size-1)//size*size for s in sequences))
                    for size in (16, 32)]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(status='complete',
        execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_revision=model['revision'], rows=rows,
        limits=['Prompt only; private generated tokens and workspace must be added.',
                'Potential exact-prefix sharing, not observed runtime cache hits or timing.',
                'All requests are from the same fixed model and finite history.',
                'No eviction; only complete blocks share; each partial block stays private.',
                'No cross-strategy quality equivalence.']), indent=2)+'\n')


if __name__ == '__main__':
    main()
