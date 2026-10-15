"""Produce small, provenance-bearing aggregates; raw prompts stay on server."""
import collections
import json
from pathlib import Path
import statistics
import sys

r=Path(__file__).resolve().parent
sys.path.insert(0,str(r/'external/agent-memory-profiling'))
from utils.eval_other_utils import calculate_metrics, post_process

def union(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    return sum(b-a for a,b in merged)

api=[json.loads(x) for x in (r/'outputs/api_trace.jsonl').read_text().splitlines()]
kernels=[json.loads(x) for x in (r/'outputs/embedding_kernels.jsonl').read_text().splitlines()]
result=dict(scope='native measured calls; no architecture performance predictions',runs=[])
for run in ['mem0_full_01','graph_full_01','graph_full_02','graph_full_03','embed_full_01','bm25_full_01','bm25_full_02']:
    folder=r/'outputs'/run
    if not folder.exists():continue
    manifests=[json.loads(p.read_text()) for p in folder.glob('*_manifest.json')]
    if not manifests:continue
    m=max(manifests,key=lambda item:item['start'])
    rows=[x for x in api if x.get('context',{}).get('run')==run]
    steps=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()] if (folder/'steps.jsonl').exists() else []
    row=dict(run=run,status=m['status'],validity=m['validity'],
        source_commit=m['source_commit'],dataset_sha256=m['dataset_sha256'],
        total_chunks=m.get('total_chunks'),total_queries=m.get('total_queries'),
        logged_error_count=len(m.get('logged_errors',[])),
        manifest_wall_seconds=m.get('end',m['start'])-m['start'],
        runner_sha256=m.get('runner_sha256'),
        answer_max_tokens=m.get('dataset_config',{}).get('generation_max_length'),
        retrieval_top_k=m.get('agent_config',{}).get('retrieve_num'),phases={})
    sidecar=folder/'runner_sha256.json'
    if row['runner_sha256'] is None and sidecar.exists():
        row['runner_sha256']=json.loads(sidecar.read_text())['sha256']
        row['runner_hash_source']='separate preserved snapshot; absent from original manifest'
    gpu_files=list(folder.glob('*_gpu.jsonl'))
    if gpu_files:
        samples=[json.loads(line) for p in gpu_files for line in p.read_text().splitlines()]
        row['gpu_energy_scope']='Gross NVML device energy over monitor interval, including idle resident services; excludes CPU, DRAM and wall-plug power. Not workload-attributed or thermal evidence.'
        row['gpu_measurements']={}
        for gpu in sorted({x['gpu'] for x in samples}):
            points=sorted([x for x in samples if x['gpu']==gpu],key=lambda x:x['time'])
            energy=[x for x in points if x.get('energy_j') is not None]
            row['gpu_measurements'][str(gpu)]=dict(
                samples=len(points),observed_seconds=points[-1]['time']-points[0]['time'],
                counter_interval_seconds=energy[-1]['time']-energy[0]['time'] if len(energy)>1 else None,
                gross_counter_energy_j=energy[-1]['energy_j']-energy[0]['energy_j'] if len(energy)>1 else None,
                sampled_peak_power_w=max(x['power_w'] for x in points),
                sampled_peak_device_memory_bytes=max(x['memory_used'] for x in points))
    if run=='mem0_full_01':
        rows=[x for x in rows if x['context'].get('index',99)<9]
        steps=[x for x in steps if x['index']<9]
        row['measurement_subset']='First 9 chunks only, before invalid references in chunk 10; no completed quality result'
    for phase in ['construction','index_finalize','query']:
        rr=[x for x in rows if x['context'].get('phase')==phase]
        ss=[x for x in steps if x['phase']==phase]
        if not rr and not ss:continue
        chat=[x for x in rr if x['path']=='chat/completions']
        embed=[x for x in rr if x['path']=='embeddings']
        prompt=sum((x.get('usage') or {}).get('prompt_tokens',0) for x in chat)
        completion=sum((x.get('usage') or {}).get('completion_tokens',0) for x in chat)
        first=[x for x in chat if x.get('first_token')]
        phase_row=dict(recorded_steps=len(ss),step_seconds=sum(x['end']-x['start'] for x in ss),
            chat_calls=len(chat),embedding_calls=len(embed),prompt_tokens=prompt,completion_tokens=completion,
            output_token_fraction=completion/(prompt+completion) if prompt+completion else None,
            chat_interval_union_seconds=union([(x['start'],x['end']) for x in chat]),
            embedding_interval_union_seconds=union([(x['start'],x['end']) for x in embed]),
            chat_sum_seconds=sum(x['end']-x['start'] for x in chat),
            embedding_api_batch_histogram=dict(collections.Counter(len(x['request']['input']) if isinstance(x['request']['input'],list) else 1 for x in embed)),
            first_token_timed_calls=len(first),
            first_token_wait_sum_seconds=sum(x['first_token']-x['start'] for x in first),
            after_first_token_sum_seconds=sum(x['end']-x['first_token'] for x in first))
        matched=[k for k in kernels if any(e['start']<=k['start'] and k['end']<=e['end'] for e in embed)]
        if matched:
            phase_row['embedding_kernel_measurements']=dict(
                matching='Kernel timestamp interval contained in a traced embedding request',
                forwards=len(matched),cuda_seconds_sum=sum(k['cuda_seconds'] for k in matched),
                actual_batch_histogram=dict(collections.Counter(k['batch'] for k in matched)),
                sequences=sum(k['batch'] for k in matched),
                unpadded_tokens=sum(sum(k['lengths']) for k in matched),
                padded_tokens=sum(k['batch']*k['padded_length'] for k in matched),
                minimum_sequence_length=min(min(k['lengths']) for k in matched),
                maximum_sequence_length=max(max(k['lengths']) for k in matched))
        if phase=='construction' and run=='mem0_full_01':
            total=phase_row['step_seconds']
            removable=phase_row['first_token_wait_sum_seconds']+phase_row['embedding_interval_union_seconds']
            phase_row['perfect_first_token_wait_and_embedding_removal_speedup_ceiling']=total/(total-removable)
            phase_row['ceiling_scope']='Sequential measured prefix only; TTFT includes queue/network/first decode. Removes MORE than prefill; not a new-engine speedup prediction.'
        row['phases'][phase]=phase_row
    answers=folder/'answers.jsonl'
    if answers.exists():
        metrics=[]
        raw_metrics=[]
        for line in answers.read_text().splitlines():
            a=json.loads(line)
            answer=a['answer'] if isinstance(a['answer'],list) else [a['answer']]
            raw_metrics.append(calculate_metrics(a['response']['output'],answer))
            processed,_=post_process(a['response'],answer,m['dataset_config'])
            metrics.append(processed)
        row['answer_metrics']=dict(count=len(metrics),metric_definition='pinned artifact LongMemEval post_process: per-metric max of raw and parsed output; lexical only, no semantic judge',
            means={k:statistics.mean(x[k] for x in metrics) for k in metrics[0]} if metrics else {},
            raw_output_means={k:statistics.mean(x[k] for x in raw_metrics) for k in raw_metrics[0]} if raw_metrics else {})
    row['stored_state_bytes']=sum(p.stat().st_size for p in (folder/'state').rglob('*') if p.is_file()) if (folder/'state').exists() else 0
    row['stored_cache_bytes']=sum(p.stat().st_size for p in (folder/'rag_cache').rglob('*') if p.is_file()) if (folder/'rag_cache').exists() else 0
    row['state_size_scope']='Serialized files only, not peak CPU working set or accelerator memory; partial failed states are not full-corpus sizes.'
    result['runs'].append(row)
(r/'outputs/run_summary.json').write_text(json.dumps(result,indent=2)+'\n')
for row in result['runs']:
    print(row['run'],row['status'],row['phases'])
