"""Conditional capacity and streaming bounds, not a chip performance simulator.

Use all seven recorded prompt shapes, with a controlled 64-token output horizon.
The 16 GiB is KV-only headroom after weights/runtime/workspaces, not total memory.
No quality equivalence, arrival process or physical implementation is assumed.
"""
import hashlib
import json
from pathlib import Path

from workload_accounting import decoder_shape


def capacity_batch(kv_budget, kv_bytes_per_token, shared_tokens, private_tokens):
    """Exact integer capacity for one shared prefix and B private suffixes."""
    if kv_budget < 0 or kv_bytes_per_token <= 0 or shared_tokens < 0 or private_tokens <= 0:
        raise ValueError('Invalid capacity or token count')
    remaining = kv_budget - kv_bytes_per_token * shared_tokens
    return max(0, remaining // (kv_bytes_per_token * private_tokens))


def stream_step_floor(weight_bytes, batch, kv_bytes_per_sequence, bandwidth):
    """Conditional lower bound: one weight pass plus per-sequence KV streaming.

    Shared-prefix capacity does NOT imply free sharing of attention reads.
    Zero compute/launch/queue overhead is a diagnostic relaxation only.
    """
    if weight_bytes <= 0 or batch < 1 or kv_bytes_per_sequence < 0 or bandwidth <= 0:
        raise ValueError('Invalid work or bandwidth')
    return (weight_bytes + batch * kv_bytes_per_sequence) / bandwidth


def main():
    root = Path(__file__).resolve().parents[1]
    names = ['evidence/architecture/finite_prefix.json',
             'evidence/architecture/representation_controls.json',
             'evidence/phase2/workload_accounting.json']
    inputs = [json.loads((root / name).read_text()) for name in names]
    records = inputs[0]['summary']['results'] + inputs[1]['summary']['results']
    assert all(a['summary']['status'] == 'complete' for a in inputs[:2])
    configs = inputs[2]['models']
    shapes = {m: decoder_shape(configs[m]['config'])
              for m in ('Qwen/Qwen3-8B', 'Qwen/Qwen3-32B')}
    kv = shapes['Qwen/Qwen3-32B']['kv_bytes_per_token']
    budget, output_tokens = 16 * 2**30, 64
    rows = []
    index_sets = []
    for strategy in ('raw', 'raw_512', 'mem0'):
        calls = [r for r in records if r['strategy'] == strategy]
        assert len(calls) == 7 and len({r['index'] for r in calls}) == 7
        assert all(not r['capped'] for r in calls)
        index_sets.append({r['index'] for r in calls})
        prompt = max(r['input_tokens'] for r in calls)
        live = prompt + output_tokens - 1
        no_share = capacity_batch(budget, kv, 0, live)
        # Explicit counterfactual: all but the last 32 input tokens are an exact
        # reusable prefix; do not infer this from semantic similarity or hashes.
        shared = prompt - 32
        shared_cap = capacity_batch(budget, kv, shared, 32 + output_tokens - 1)
        rows.append(dict(strategy=strategy, input_tokens=[r['input_tokens'] for r in calls],
                         max_prompt_tokens=prompt, controlled_live_tokens=live,
                         kv_bytes_per_sequence=kv * live,
                         no_sharing_capacity_batch=no_share,
                         hypothetical_shared_prefix_tokens=shared,
                         hypothetical_shared_capacity_batch=shared_cap,
                         capacity_and_ready_caps=[dict(ready=ready,
                             independent_prefix=min(ready, no_share),
                             hypothetical_shared_prefix=min(ready, shared_cap))
                             for ready in (1, 8, 32)]))
    assert index_sets[0] == index_sets[1] == index_sets[2]
    result = dict(
        scope=__doc__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decoder_shape_source_sha256=hashlib.sha256(
            (root / 'workload_accounting.py').read_bytes()).hexdigest(),
        input_sha256={n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
        controls=dict(kv_only_headroom_bytes=budget, output_horizon_tokens=output_tokens,
                      sizing='Maximum of all seven recorded prompt lengths per representation',
                      ready_counts='Controlled independent ready requests, not observed arrivals',
                      prefix_sharing='Counterfactual exact prefix; actual reusable length unmeasured',
                      capacity='Only a necessary constraint, not achievable batch or throughput'),
        capacity_rows=rows,
        weight_only_bandwidth_requirements=[
            dict(model=model, weight_stream_bytes=s['streamed_linear_weight_bytes'],
                 step_budget_ms=ms,
                 required_effective_TB_s=s['streamed_linear_weight_bytes'] / (ms / 1000) / 1e12)
            for model, s in shapes.items() for ms in (50, 100, 200)],
        assumptions=['Dense BF16 autoregressive generation, one model-weight pass per decode step',
                     'Linear streaming weights include output head, exclude input embedding lookup',
                     'No persistent reuse of streamed weights between decode steps',
                     'No speculative multi-token acceptance; bandwidth values are effective, not peak',
                     'No compute, prefill, queue, energy, quality or hardware-cost prediction'])
    target = root / 'evidence/architecture/capacity_latency_bounds.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(row['strategy'], row['max_prompt_tokens'],
              row['no_sharing_capacity_batch'], row['hypothetical_shared_capacity_batch'])
    plot(root, shapes['Qwen/Qwen3-32B']['streamed_linear_weight_bytes'])


def plot(root, weight_bytes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    batch = list(range(1, 65))
    for bandwidth, label in ((0.64e12, '0.64 TB/s'), (1.28e12, '1.28 TB/s')):
        # Weight-only relaxation intentionally removes KV to expose the floor.
        time = [stream_step_floor(weight_bytes, b, 0, bandwidth) for b in batch]
        axes[0].plot(batch, [1000 * t for t in time], label=label)
        axes[1].plot(batch, [b / t for b, t in zip(batch, time)], label=label)
    axes[0].set(ylabel='Weight-pass time floor (ms)', xlabel='Batch', ylim=(0, 115))
    axes[1].set(ylabel='Weight-only throughput ceiling (token/s)', xlabel='Batch')
    axes[0].set_title('Per-request dependency remains')
    axes[1].set_title('More requests share each pass')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle('Qwen3-32B BF16: conditional streaming bounds, not measured performance', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(root / f'evidence/architecture/capacity_latency_bounds.{ext}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
