"""Validate frozen A800 results and compare exactly the same request sets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'evidence/architecture/capacity_batch.json'
    data = json.loads(path.read_text())
    manifest, cases = data['manifest'], data['cases']
    assert data['status'] == manifest['status'] == 'complete'
    assert manifest['runner_sha256'] == hashlib.sha256(
        (ROOT / 'architecture/capacity_batch_probe.py').read_bytes()).hexdigest()
    steps = manifest['args']['steps']
    assert steps == 16 and len(cases) == manifest['planned_cases'] == 33
    indices = [3, 28, 30, 31, 39, 48, 54]
    kv_per_token = 262144  # 64 layers * 2 K/V * 8 heads * 128 dim * BF16.
    assert manifest['kv_bytes_per_token'] == kv_per_token
    comparisons, chunks, checks, skipped = [], [], [], []
    for strategy in ('raw', 'raw_512', 'mem0'):
        group = [r for r in cases if r['strategy'] == strategy]
        assert len(group) == 11
        refs = {r['indices'][0]: r for r in group if r['kind'] == 'single_reference'}
        assert sorted(refs) == indices
        assert all(r['status'] == 'complete' for r in refs.values())
        for row in group:
            batch, length = row['batch'], row['padded_input_tokens']
            assert batch == len(row['indices']) == len(row['actual_input_tokens'])
            assert length == max(r['actual_input_tokens'][0] for r in refs.values())
            for index, tokens in zip(row['indices'], row['actual_input_tokens']):
                receipt = next(s for s in manifest['source_receipts']
                               if s['index'] == index and s['strategy'] == strategy)
                assert receipt['input_tokens'] == tokens
            kv = batch * (length + steps) * kv_per_token
            required = manifest['weight_bytes'] + kv + manifest['args']['reserve_gib'] * 2**30
            if row['status'] == 'capacity_skipped':
                assert required > manifest['gpu_memory_bytes']
                assert row['projected_kv_bytes'] == kv
                skipped.append(dict(strategy=strategy, batch=batch,
                                    weights_plus_kv_gib=(manifest['weight_bytes'] + kv) / 2**30))
                continue
            assert row['status'] == 'complete' and required <= manifest['gpu_memory_bytes']
            assert len(row['decode_cuda_seconds']) == steps
            assert min(row['decode_cuda_seconds']) > 0 and row['prefill_cuda_seconds'] > 0
            assert row['prefill_kv_bytes'] == batch * length * kv_per_token
            assert row['final_kv_bytes'] == kv
            assert row['peak_allocated_bytes'] >= row['allocated_bytes']
            assert row['allocated_bytes'] >= manifest['weight_bytes'] + kv
            seconds = sum(row['decode_cuda_seconds'])
            assert abs(row['mean_decode_step_ms'] - 1000 * seconds / steps) < 1e-8
            assert abs(row['forward_decode_tokens_per_second'] - batch * steps / seconds) < 1e-8
            checks.extend(row['numerical_checks'])
            if row['kind'] == 'distinct_batch':
                assert row['indices'] == indices[:batch]
                matched = [refs[i] for i in row['indices']]
                serial_decode = sum(sum(r['decode_cuda_seconds']) for r in matched)
                serial_prefill = sum(r['prefill_cuda_seconds'] for r in matched)
                comparisons.append(dict(strategy=strategy, batch=batch, indices=row['indices'],
                    numerical_pass=row['numerical_pass'],
                    matched_single_mean_step_ms=1000 * serial_decode / (steps * batch),
                    matched_single_step_range_ms=[min(r['mean_decode_step_ms'] for r in matched),
                                                 max(r['mean_decode_step_ms'] for r in matched)],
                    batch_mean_step_ms=row['mean_decode_step_ms'],
                    serial_forward_tps=batch * steps / serial_decode,
                    batch_forward_tps=batch * steps / seconds,
                    decode_throughput_ratio=serial_decode / seconds,
                    per_request_step_ratio=row['mean_decode_step_ms'] /
                                           (1000 * serial_decode / (steps * batch)),
                    prefill_serial_to_batch_ratio=serial_prefill / row['prefill_cuda_seconds'],
                    prefill_plus_decode_serial_to_batch_ratio=(serial_prefill + serial_decode) /
                                                              (row['prefill_cuda_seconds'] + seconds),
                    kv_gib=kv / 2**30, peak_allocated_gib=row['peak_allocated_bytes'] / 2**30))
            elif row['kind'] == 'unchunked_control':
                ref = refs[row['indices'][0]]
                chunks.append(dict(strategy=strategy, index=row['indices'][0],
                    numerical_pass=row['numerical_pass'],
                    chunked_prefill_seconds=ref['prefill_cuda_seconds'],
                    unchunked_prefill_seconds=row['prefill_cuda_seconds'],
                    chunked_peak_gib=ref['peak_allocated_bytes'] / 2**30,
                    unchunked_peak_gib=row['peak_allocated_bytes'] / 2**30))
    assert len(checks) == manifest['numerical_checks'] == 62
    assert all(c['numerical_pass'] == (c['relative_logit_l2'] <= 0.02 and
                                     c['max_absolute_logit_error'] <= 0.5) for c in checks)
    assert {(r['strategy'], r['batch']) for r in skipped} == {('raw', 4), ('raw', 7)}
    assert manifest['completed_cases'] == 31 and len(comparisons) == 7
    result = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  comparisons=comparisons, chunk_controls=chunks, capacity_skips=skipped,
                  validation=dict(completed_cases=31, skipped_cases=2, numerical_checks=len(checks),
                    numerical_passes=sum(c['numerical_pass'] for c in checks),
                    argmax_agreements=sum(c['argmax_agreement'] for c in checks),
                    max_relative_logit_l2=max(c['relative_logit_l2'] for c in checks),
                    max_absolute_logit_error=max(c['max_absolute_logit_error'] for c in checks)),
                  limits=['HF eager SDPA/DynamicCache calibration, not production serving.',
                          'Batch and single attention dispatch can differ with padding masks.',
                          'Same request set, same recorded teacher-forced 16-token work.',
                          'No cross-representation quality equivalence, no new hardware result.'])
    target = ROOT / 'evidence/architecture/capacity_batch_analysis.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    for row in comparisons:
        print(row['strategy'], row['batch'], round(row['decode_throughput_ratio'], 3),
              round(row['per_request_step_ratio'], 3))
    print(result['validation'])


if __name__ == '__main__':
    main()
