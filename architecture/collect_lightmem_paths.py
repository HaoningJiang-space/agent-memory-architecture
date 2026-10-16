"""Export counts/shapes from bounded path controls; never export text or keys."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from architecture.collect_family import ROOT, collect, digest, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='lightpath2')
    parser.add_argument('--services', default='lightpath_services')
    parser.add_argument('--out', default='outputs/lightmem_paths_public.json')
    args = parser.parse_args()
    service_path = ROOT / 'outputs' / args.services
    result = dict(scope='Three single-run LightMem path controls, not paper reproduction or quality equivalence.',
                  collector_sha256=digest(Path(__file__)), cases=[])
    for variant, stm in [('flat', 512), ('full', 512), ('full', 768)]:
        session = f'{args.prefix}_{variant}{stm}'
        path = ROOT / 'outputs' / session / 'lightmem'
        manifest = json.loads((path / 'manifest.json').read_text())
        public = collect(path, service_path)
        public.update(session=session, variant=variant, stm_tokens=stm,
            records_before_maintenance=manifest['records_before_maintenance'],
            message_conservation=manifest['message_conservation'],
            flush_adapter=manifest['flush_adapter'], sensory_tokens=manifest['sensory_tokens'],
            native_tokens=manifest['native_tokens'],
            initialization_seconds=manifest['initialization_end']-manifest['start'],
            after_initialization_wall_seconds=manifest['end']-manifest['initialization_end'],
            encoder_linears=manifest['encoder_linears'])
        public['effective_path'] = {k: manifest['effective_config'][k] for k in
            ['pre_compress', 'topic_segment', 'precomp_topic_shared', 'extraction_mode', 'messages_use', 'update']}
        public['encoder_source'] = {k: v for k, v in manifest['encoder_source'].items() if k != 'path'}
        public['encoder'] = {k: manifest.get('encoder', {}).get(k) for k in
            ['parameters', 'parameter_bytes', 'dtype']}
        encoder = path / 'encoder_forwards.jsonl'
        forwards = records(encoder) if encoder.exists() else []
        if variant == 'full':
            assert forwards and manifest['message_conservation']['equal']
            assert not (path / 'encoder_errors.jsonl').exists()
            public['artifact_sha256'][encoder.name] = digest(encoder)
        else:
            assert not forwards and not manifest['encoder_linears']
        public['encoder_forwards'] = forwards
        public['encoder_stages'] = []
        for stage in sorted({r['stage'] for r in forwards}):
            rows = [r for r in forwards if r['stage'] == stage]
            linears = [r for r in manifest['encoder_linears'] if r['stage'] == stage]
            shapes = Counter(tuple(r['shape']) for r in rows)
            public['encoder_stages'].append(dict(stage=stage, calls=len(rows),
                summed_cuda_seconds=sum(r['cuda_seconds'] for r in rows),
                shapes=[dict(shape=list(s), count=n) for s,n in sorted(shapes.items())],
                linear_flops=sum(r['flops'] for r in linears),
                one_pass_linear_weight_bytes=sum(r['one_pass_weight_bytes'] for r in linears)))
        # A digest permits same-prompt checks without publishing private text.
        raw_calls = records(path / 'llm_private.jsonl')
        public['prompt_sha256'] = [dict(id=r['id'], sha256=hashlib.sha256(
            json.dumps(r['request']['messages'], sort_keys=True).encode()).hexdigest()) for r in raw_calls]
        result['cases'].append(public)
    assert len({r['history_sha256'] for r in result['cases']}) == 1
    result['service_configuration'] = {k: dict(argv=v['argv'], environment=v['environment']) for k,v in
        json.loads((ROOT/f'services_{args.services}.json').read_text()).items()}
    result['limits'] = ['Counts/structures checked, not task correctness or fact-retention quality.',
        'Three probes may not be answerable from this finite history.',
        'Same flush adapter applied to both arms; old upstream stale-vector update behavior retained.',
        'Prefix-cache reset requested between cases on isolated vLLM services; within-case cache remains enabled.',
        'Encoder CUDA events and hooks are intrusive; end-to-end time includes Python/CPU/server work.',
        'No energy sampling, production arrivals, continuous write/read contention or SLO measurement.',
        'FP32 encoder versus BF16 8B/32B; encoder linear counts exclude attention/softmax/nonlinear FLOPs.',
        'Message conservation refers to compressed messages delivered to extraction, not semantic information conservation.']
    (ROOT/args.out).write_text(json.dumps(result, indent=2)+'\n')
    for case in result['cases']:
        print(case['session'], case['records'], case['after_initialization_wall_seconds'],
              [(r['phase'], r['count'], r['total_input_tokens'], r['total_output_tokens']) for r in case['phase_stats']],
              [(r['stage'], r['calls'], r['summed_cuda_seconds']) for r in case['encoder_stages']])


if __name__ == '__main__':
    main()
