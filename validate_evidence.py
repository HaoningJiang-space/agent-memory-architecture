"""Cross-check finished baseline outputs against independent raw API records.

This validates completion and accounting, not semantic answer correctness.
"""
import collections
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
output = root / 'outputs'
api = [json.loads(s) for s in (output / 'api_trace.jsonl').read_text().splitlines()]
assert len({row['id'] for row in api}) == len(api), 'Duplicate trace IDs'
summary = json.loads((output / 'run_summary.json').read_text())
results = []
for name in ['embed_full_01', 'bm25_full_02']:
    folder = output / name
    manifest = json.loads((folder / 'both_manifest.json').read_text())
    answers = [json.loads(s) for s in (folder / 'answers.jsonl').read_text().splitlines()]
    steps = [json.loads(s) for s in (folder / 'steps.jsonl').read_text().splitlines()]
    calls = [row for row in api if row.get('context', {}).get('run') == name]
    assert manifest['status'] == 'complete' and manifest['validity'] == 'valid'
    assert not manifest['logged_errors']
    assert manifest['dataset_sha256'] == hashlib.sha256((root / 'longmemeval.json').read_bytes()).hexdigest()
    assert manifest['source_commit'] == '30a157b7b49f732ee4b961cb19d7f877cf475f81'
    assert len(answers) == manifest['total_queries'] == 60
    assert [a['index'] for a in answers] == list(range(60))
    assert all(a['response']['output'].strip() for a in answers)
    phases = collections.Counter(s['phase'] for s in steps)
    assert phases == {'construction': 88, 'index_finalize': 1, 'query': 60}
    assert all(s['end'] >= s['start'] for s in steps)
    assert all(a['end'] <= b['start'] for a, b in zip(steps, steps[1:])), 'Native steps must be sequential'
    for call in calls:
        assert call['status'] == 200 and not call.get('error')
        assert call['start'] <= call['end']
        assert call.get('usage')
        if call['path'] == 'chat/completions':
            assert call['model'] == 'Qwen/Qwen3-32B'
            assert call['context']['phase'] == 'query'
            reason = call.get('finish_reason')
            if reason is None:
                reason = call['response']['choices'][0]['finish_reason']
            assert reason == 'stop', 'Incomplete answer'
        else:
            assert call['path'] == 'embeddings' and call['model'] == 'Qwen/Qwen3-Embedding-0.6B'
            assert call['embedding_count'] > 0
    query_calls = [c for c in calls if c['path'] == 'chat/completions']
    assert len(query_calls) == 60
    assert sorted(c['context']['index'] for c in query_calls) == list(range(60))
    row = next(s for s in summary['runs'] if s['run'] == name)
    assert row['answer_metrics']['count'] == 60
    for phase, aggregate in row['phases'].items():
        chats = [c for c in calls if c['context']['phase'] == phase and c['path'] == 'chat/completions']
        assert aggregate['prompt_tokens'] == sum(c['usage']['prompt_tokens'] for c in chats)
        assert aggregate['completion_tokens'] == sum(c['usage']['completion_tokens'] for c in chats)
    results.append(dict(run=name,valid=True,chunks=88,answers=60,api_calls=len(calls),
                        failure_count=0,semantic_quality='Not validated by this integrity check'))

assert next(x for x in summary['runs'] if x['run']=='mem0_full_01')['validity']=='invalid'
assert next(x for x in summary['runs'] if x['run']=='graph_full_03')['validity']=='invalid'
assert next(x for x in summary['runs'] if x['run']=='bm25_full_01')['validity']=='invalid'
report = dict(scope='Independent artifact integrity and completion checks, not semantic evaluation',
              checks_passed=True, runs=results)
(output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
