"""Account for logical work from frozen real requests; never call it DRAM traffic."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def decoder_shape(config, scalar_bytes=2):
    layers = config['num_hidden_layers']
    hidden = config['hidden_size']
    query = config['num_attention_heads'] * config['head_dim']
    key_value = config['num_key_value_heads'] * config['head_dim']
    intermediate = config['intermediate_size']
    # Q, K, V, O and gate/up/down projections; input embedding is a lookup.
    projection_parameters = layers * (2*hidden*query + 2*hidden*key_value +
                                       3*hidden*intermediate)
    head_parameters = hidden * config['vocab_size']
    return dict(layers=layers, query_width=query,
                projection_parameters=projection_parameters,
                head_parameters=head_parameters,
                streamed_linear_weight_bytes=(projection_parameters+head_parameters)*scalar_bytes,
                kv_bytes_per_token=2*layers*key_value*scalar_bytes)


def request_work(config, prompt, completion):
    """First output uses prefill logits; only completion-1 decode forwards remain."""
    assert prompt > 0 and completion >= 1
    shape = decoder_shape(config)
    steps = completion - 1
    context_sum = steps*prompt + steps*(steps+1)//2
    prefill_flops = (2*shape['projection_parameters']*prompt +
                     2*shape['head_parameters'] +
                     2*shape['layers']*shape['query_width']*prompt*(prompt+1))
    decode_flops = (2*(shape['projection_parameters']+shape['head_parameters'])*steps +
                    4*shape['layers']*shape['query_width']*context_sum)
    return dict(prompt_tokens=prompt, completion_tokens=completion, decode_steps=steps,
                prefill_logical_flops=prefill_flops, decode_logical_flops=decode_flops,
                prefill_weight_stream_bytes=shape['streamed_linear_weight_bytes'],
                decode_weight_stream_bytes_B1=shape['streamed_linear_weight_bytes']*steps,
                decode_kv_logical_read_bytes=shape['kv_bytes_per_token']*context_sum,
                kv_logical_write_bytes=shape['kv_bytes_per_token']*(prompt+steps),
                peak_request_kv_bytes=shape['kv_bytes_per_token']*(prompt+steps))


def main():
    root = Path(__file__).resolve().parent
    output = root/'outputs/phase2'
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((root/'outputs/raw_artifact_sha256.json').read_text())
    source = root/'outputs/api_trace.jsonl'
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == receipt['files']['outputs/api_trace.jsonl']['sha256']
    models = json.loads((root/'models.json').read_text())
    configs = {name: json.loads((Path(info['path'])/'config.json').read_text())
               for name, info in models.items()}
    rows = [json.loads(line) for line in raw.splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        context = row.get('context', {})
        if row.get('path') != 'chat/completions' or not row.get('usage'):
            continue
        run = context.get('run')
        if run == 'mem0_full_01' and context.get('phase') == 'construction' and context['index'] < 9:
            group = 'mem0_update_prefix' if len(row['request']['messages']) == 1 else 'mem0_extract_prefix'
        elif run in ('embed_full_01', 'bm25_full_02') and context.get('phase') == 'query':
            group = run+'_query'
        else:
            continue
        usage = row['usage']
        work = request_work(configs[row['model']], usage['prompt_tokens'], usage['completion_tokens'])
        work.update(trace_id=row['id'], model=row['model'],
                    request_seconds=row['end']-row['start'],
                    ttft_seconds=row['first_token']-row['start'],
                    after_first_token_seconds=row['end']-row['first_token'])
        grouped[group].append(work)
    report = dict(source_trace_sha256=digest, models={name:dict(config=cfg,
                  shape=decoder_shape(cfg) if 'Embedding' not in name else None,
                  revision=models[name]['revision']) for name, cfg in configs.items()},
                  scope='Logical operation counts and explicit dataflow scenarios, not measured hardware traffic. '
                        'Mem0 is the first nine structurally completed chunks of a failed run; '
                        'not a full or quality-validated memory system. Timings include serving overhead.', groups={})
    for name, calls in grouped.items():
        totals = {key: sum(call[key] for call in calls) for key in calls[0]
                  if key not in ('trace_id', 'model', 'peak_request_kv_bytes')}
        totals.update(calls=len(calls), max_request_kv_bytes=max(c['peak_request_kv_bytes'] for c in calls),
                      output_token_share=totals['completion_tokens']/(totals['prompt_tokens']+totals['completion_tokens']),
                      after_first_time_share=totals['after_first_token_seconds']/totals['request_seconds'])
        report['groups'][name] = dict(totals=totals, calls=calls)
    # This is a normalized limiting case, not an A800 or prospective NPU prediction.
    report['token_share_counterexample'] = [
        dict(output_token_share=fraction, ridge_flops_per_byte=ridge, batch=batch,
             decode_to_prefill_time_ratio=(fraction/(1-fraction))*ridge/batch,
             assumptions='BF16, weight-dominated decode, compute-saturated prefill, '
                         'perfect weight reuse across independent same-model decode requests; '
                         'ignore attention/launch/queueing and first-token correction')
        for fraction in (0.009, 0.046, 0.10, 0.285)
        for ridge in (50, 100, 200) for batch in (1, 4, 16, 64)]
    target = output/'workload_accounting.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({name:value['totals'] for name,value in report['groups'].items()}, indent=2))


if __name__ == '__main__':
    main()
