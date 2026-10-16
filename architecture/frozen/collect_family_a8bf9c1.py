"""Export numeric family evidence on moe-server; never export source text."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def overlap(rows):
    """Half-open intervals: end precedes start when two calls touch."""
    events = sorted([(r['start'], 1) for r in rows] + [(r['end'], -1) for r in rows])
    active = peak = 0
    union = 0
    previous = events[0][0] if events else 0
    for time, delta in events:
        if active: union += time - previous
        active += delta
        assert active >= 0
        peak = max(peak, active)
        previous = time
    assert active == 0
    return dict(api_concurrency_peak=peak, api_interval_union_seconds=union,
                summed_api_seconds=sum(r['end'] - r['start'] for r in rows))


def metrics(text):
    result = {}
    for line in text.splitlines():
        if line.startswith('vllm:iteration_tokens_total_bucket'):
            result[re.search(r'le="([^"]+)"', line).group(1)] = float(line.rsplit(' ', 1)[1])
    return result


def collect(path, service_path):
    manifest = json.loads((path/'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    public = {k: manifest[k] for k in ['algorithm', 'status', 'start', 'end', 'runner_sha256', 'dataset_sha256',
        'history_sha256', 'source_sha256', 'native_revision', 'records', 'versions', 'state_disk_bytes',
        'selection', 'quality_scope', 'deployment']}
    public['artifact_sha256'] = {p.name: digest(p) for p in [path/'manifest.json', path/'llm_private.jsonl',
        path/'operations.jsonl', path/'state_private.json']}
    calls = []
    for row in records(path/'llm_private.jsonl'):
        assert row['validated']
        response, request = row['response'], row['request']
        assert response['choices'][0]['finish_reason'] == 'stop'
        calls.append(dict(id=row['id'], phase=row['phase'], model=request['model'],
            L=response['usage']['prompt_tokens'], O=response['usage']['completion_tokens'],
            start=row['start'], end=row['end'], finish_reason='stop'))
    api = [r for r in records(service_path/'api_trace.jsonl') if
        r.get('context', {}).get('algorithm') == manifest['algorithm'] and
        manifest['start'] <= r['start'] < manifest['end'] and r['path'] == 'embeddings']
    assert all(r['status'] == 200 for r in api)
    embedding = [dict(id=r['id'], phase=r['context']['phase'], start=r['start'], end=r['end'],
        sequences=r['embedding_count'], tokens=r['usage']['prompt_tokens']) for r in api]
    kernels = [r for r in records(service_path/'embedding_kernels.jsonl')
               if manifest['start'] <= r['start'] < manifest['end']]
    assert sum(k['batch'] for k in kernels) == sum(r['sequences'] for r in embedding)
    assert sum(sum(k['lengths']) for k in kernels) == sum(r['tokens'] for r in embedding)
    for k in kernels:
        k['linear_M_padded'] = k['batch'] * k['padded_length']
        matches = [r for r in embedding if r['start'] <= k['start'] and r['end'] >= k['end']]
        assert matches
        phases = {r['phase'] for r in matches}
        assert len(phases) == 1
        k['phase'] = phases.pop()
    stats = []
    for phase in sorted({r['phase'] for r in calls}):
        for model in sorted({r['model'] for r in calls if r['phase'] == phase}):
            rows = [r for r in calls if r['phase'] == phase and r['model'] == model]
            stats.append(dict(phase=phase, model=model, count=len(rows),
                total_input_tokens=sum(r['L'] for r in rows), total_output_tokens=sum(r['O'] for r in rows),
                mean_L=sum(r['L'] for r in rows)/len(rows), mean_O=sum(r['O'] for r in rows)/len(rows),
                **overlap(rows)))
    public.update(calls=calls, phase_stats=stats, embedding_calls=embedding, embedding_kernels=kernels,
                  operations=records(path/'operations.jsonl'))
    before, after = path/'engine_metrics_before.json', path/'engine_metrics_after.json'
    if before.exists() and after.exists():
        a, b = json.loads(before.read_text()), json.loads(after.read_text())
        histogram = {}
        for port in a:
            first, last = metrics(a[port]), metrics(b[port])
            delta = {key: last[key]-first[key] for key in first}
            assert all(v >= 0 for v in delta.values())
            histogram[port] = delta
        public['engine_output_token_buckets'] = histogram
    public['batch_boundary'] = 'API concurrency is not kernel batch. Embedding M is observed padded B*L. vLLM 0.10.2 v1 metrics/stats.py adds full prompt_len on first output; its iteration histogram is output-accounted tokens, NOT scheduled tokens or padded kernel M.'
    return public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simple', default='family_02/simplemem')
    parser.add_argument('--light', default='family_01/lightmem')
    parser.add_argument('--out', default='outputs/family_01/public_summary.json')
    args = parser.parse_args()
    result = dict(scope='Bounded native workload-family coverage; not complete applications or equal-quality ranking.',
        collector_sha256=digest(Path(__file__)), algorithms=[
            collect(ROOT/'outputs'/args.simple, ROOT/'outputs/family_01'),
            collect(ROOT/'outputs'/args.light, ROOT/'outputs/family_01')])
    assert result['algorithms'][0]['history_sha256'] == result['algorithms'][1]['history_sha256']
    failed = ROOT/'outputs/family_01/simplemem/manifest.json'
    result['excluded_attempt'] = dict(path=str(failed.parent), manifest_sha256=digest(failed),
        status=json.loads(failed.read_text())['status'],
        reason='48 entries generated; native FTS index unavailable because pylance dependency was absent. Whole attempt excluded from successful-path summary.')
    (ROOT/args.out).write_text(json.dumps(result, indent=2)+'\n')
    print([(a['algorithm'],len(a['calls']),a['records']) for a in result['algorithms']])


if __name__ == '__main__':
    main()
