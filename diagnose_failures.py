"""Read-only cause analysis; export counts and ID bindings, never raw histories."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
receipt=json.loads((root/'outputs/raw_artifact_sha256.json').read_text())
source_hashes={}
for name in ['outputs/api_trace.jsonl','outputs/failed_update_replay.json']:
    digest=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert digest==receipt['files'][name]['sha256'],'Frozen source changed'
    source_hashes[name]=digest
api=[json.loads(s) for s in (root/'outputs/api_trace.jsonl').read_text().splitlines()]

def content(call):
    return call.get('response_text') or call.get('response',{}).get('choices',[{}])[0].get('message',{}).get('content','')

def old_memory(call):
    prompt=call['request']['messages'][0]['content']
    marker='Below is the current content of my memory which I have collected till now.'
    return ast.literal_eval(prompt[prompt.index(marker):].split('```')[1].strip())

def bindings(call,operations):
    old=old_memory(call)
    prompt=call['request']['messages'][0]['content']
    marker='Below is the current content of my memory which I have collected till now.'
    new_facts=ast.literal_eval(prompt[prompt.index(marker):].split('```')[3].strip())
    by_text=collections.defaultdict(list)
    for item in old:
        by_text[item['text']].append(str(item['id']))
    valid={str(x['id']) for x in old}
    records=[]
    for i,op in enumerate(operations):
        if op.get('event') not in ['DELETE','UPDATE','NONE']:
            continue
        exact=by_text.get(op.get('text'),[])
        if str(op.get('id')) not in valid or (exact and str(op.get('id')) not in exact):
            records.append(dict(output_position=i,id=op.get('id'),event=op.get('event'),
                id_in_old_set=str(op.get('id')) in valid,exact_text_old_ids=exact,
                exact_text_new_fact_positions=[j for j,fact in enumerate(new_facts) if fact==op.get('text')]))
    return dict(existing_count=len(old),operation_count=len(operations),
                all_ids_equal_output_positions=all(str(op.get('id'))==str(i) for i,op in enumerate(operations)),
                event_counts=dict(collections.Counter(x.get('event') for x in operations)),
                duplicate_ids={k:v for k,v in collections.Counter(str(x.get('id')) for x in operations).items() if v>1},
                suspicious_bindings=records,
                scope='Exact text/ID matches only; UPDATE text may legitimately change. Not a semantic memory judge.')

prefix=[x for x in api if x.get('context',{}).get('run')=='mem0_full_01'
        and x['context'].get('index',99)<9 and x['path']=='chat/completions']
stages=collections.defaultdict(list)
update_counts=collections.Counter()
no_op_chars=all_record_chars=0
for index in range(9):
    calls=sorted([x for x in prefix if x['context']['index']==index],key=lambda x:x['start'])
    assert len(calls)==2
    stages['extraction'].append(calls[0]);stages['update'].append(calls[1])
    operations=json.loads(content(calls[1]))['memory']
    update_counts.update(x['event'] for x in operations)
    for op in operations:
        n=len(json.dumps(op,ensure_ascii=False))
        all_record_chars+=n
        if op['event']=='NONE':no_op_chars+=n
stage_summary={name:dict(calls=len(calls),prompt_tokens=sum(c['usage']['prompt_tokens'] for c in calls),
    completion_tokens=sum(c['usage']['completion_tokens'] for c in calls),
    seconds=sum(c['end']-c['start'] for c in calls),
    first_token_wait_seconds=sum(c['first_token']-c['start'] for c in calls)) for name,calls in stages.items()}
failed=next(x for x in api if x['id']=='bcabd788-960a-41e7-88b6-ea2db565d679')
replay=json.loads((root/'outputs/failed_update_replay.json').read_text())
mem=dict(prefix_stage_totals=stage_summary,update_event_counts=dict(update_counts),
    none_record_character_fraction=no_op_chars/all_record_chars,
    character_fraction_scope='Serialized individual records; descriptive proxy, not generated token fraction',
    failed_request=bindings(failed,json.loads(content(failed))['memory']),
    replay_bindings=[dict(model=t['model'],repeat=t['repeat'],**bindings(failed,t['response']['memory'])) for t in replay['trials']])

def partial_list(text):
    # Count only fully decoded string items before a possible truncation.
    if '[' not in text:return []
    tail=text[text.index('[')+1:];items=[];decoder=json.JSONDecoder()
    while tail:
        tail=tail.lstrip(' \t\r\n,')
        if tail.startswith(']'):break
        try:item,used=decoder.raw_decode(tail)
        except json.JSONDecodeError:break
        if not isinstance(item,str):break
        items.append(item);tail=tail[used:]
    return items

graph=[]
all_graph=[x for x in api if x.get('context',{}).get('run')=='graph_full_03' and x['path']=='chat/completions']
for call in all_graph:
    choice=call.get('response',{}).get('choices',[{}])[0]
    if choice.get('finish_reason')!='length':continue
    value=content(call);items=partial_list(value);counts=collections.Counter(items)
    longest=current=0;prior=None
    for item in items:
        current=current+1 if item==prior else 1
        longest=max(longest,current);prior=item
    graph.append(dict(trace_id=call['id'],prompt_tokens=call['usage']['prompt_tokens'],
        completion_tokens=call['usage']['completion_tokens'],complete_string_items=len(items),
        unique_string_items=len(counts),maximum_item_repetitions=max(counts.values(),default=0),
        longest_consecutive_repeat=longest,response_sha256=hashlib.sha256(value.encode()).hexdigest()))
lengths=sorted(x['usage']['completion_tokens'] for x in all_graph)
result=dict(scope='Read-only analysis of frozen requests and outputs; no new model inference',mem0=mem,
    source_sha256=source_hashes,
    graph=dict(calls=len(all_graph),completion_tokens_median=statistics.median(lengths),
        completion_tokens_p95=lengths[int(.95*(len(lengths)-1))],truncated_outputs=graph))
(root/'outputs/failure_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
