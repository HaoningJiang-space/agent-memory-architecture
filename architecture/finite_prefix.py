"""Remote-only finite-history pairing; raw text and states never leave the host.

Reuse the first 19 structurally accepted Mem0 writes, restore only the copied
index, and select ALL questions whose annotated answer-bearing turns are in
that prefix. HF eager/SDPA measurements expose live tensors, not a vLLM SLA.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str)+'\n')
    temp.replace(path)


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def norm(text):
    return ' '.join(text.split())


def support_turns(value):
    if isinstance(value, dict):
        if value.get('has_answer'):
            yield value['content']
    elif isinstance(value, list):
        for item in value:
            yield from support_turns(item)


def audit(source, creator):
    manifest = json.loads((source/'both_manifest.json').read_text())
    steps = lines(source/'steps.jsonl')
    count = next(x['index'] for x in steps if x['validation'] != 'passed')
    assert count == 19 and [x['index'] for x in steps if x['validation']=='passed'] == list(range(count))
    cutoff = steps[count]['start']
    validations = lines(source/'protocol_validation.jsonl')
    assert all(any(v['context']['index']==i and v['accepted'] for v in validations) for i in range(count))
    mutations = [x for x in lines(source/'index.jsonl') if x['operation'] in ('insert','update','delete')
                 and x['context']['index'] >= count]
    # This snapshot has one failed-step ADD. Do not generalize rollback silently.
    assert len(mutations)==1 and mutations[0]['operation']=='insert'
    connection = sqlite3.connect('file:'+str(source/'state/mem0/history.db')+'?mode=ro',uri=True)
    connection.row_factory = sqlite3.Row
    history = [dict(x) for x in connection.execute('select * from history order by rowid')]
    connection.close()
    state, excluded = {}, []
    for index,row in enumerate(history):
        timestamp = row['updated_at'] or row['created_at']
        if timestamp is None:
            # Native DELETE history omits both timestamps. Require surrounding
            # committed SQLite row order to bracket it inside the valid prefix.
            assert row['event']=='DELETE' and 0 < index < len(history)-1
            bounds=[history[j]['updated_at'] or history[j]['created_at'] for j in (index-1,index+1)]
            assert all(t and datetime.fromisoformat(t).timestamp()<cutoff for t in bounds)
            stamp=datetime.fromisoformat(bounds[1]).timestamp()
        else:
            stamp = datetime.fromisoformat(timestamp).timestamp()
        if stamp >= cutoff:
            excluded.append(row)
        elif row['event'] in ('ADD','UPDATE'):
            state[row['memory_id']] = row['new_memory']
        elif row['event']=='DELETE':
            del state[row['memory_id']]
        else:
            raise ValueError(row['event'])
    assert len(excluded)==1 and excluded[0]['event']=='ADD' and excluded[0]['memory_id'] not in state
    assert mutations[0]['start']-.1 <= datetime.fromisoformat(excluded[0]['created_at']).timestamp() <= mutations[0]['end']
    data_path = ROOT/'longmemeval.json'
    assert sha(data_path)==manifest['dataset_sha256']
    data = json.loads(data_path.read_text())[0]
    chunks = creator.get_chunks()[0][:count]
    prefix = norm(' '.join(chunks))
    eligible = []
    for i,question in enumerate(data['questions']):
        support = list(support_turns(data['metadata']['haystack_sessions'][i]))
        if support and all(norm(turn) in prefix for turn in support):
            eligible.append(dict(index=i,question=question,answer=data['answers'][i],
                                 date=data['metadata']['question_dates'][i],
                                 support_sha256=[hashlib.sha256(norm(t).encode()).hexdigest() for t in support]))
    assert eligible
    receipt = dict(source_manifest_sha256=sha(source/'both_manifest.json'),dataset_sha256=sha(data_path),
                   source_files={str(p.relative_to(source)):sha(p) for p in sorted((source/'state').rglob('*')) if p.is_file()},
                   prefix_chunks=count,prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                   prefix_chars=len(prefix),accepted_state_records=len(state),excluded_failed_adds=len(excluded),
                   excluded_history_ids=[x['id'] for x in excluded],
                   eligible_questions=[dict(index=q['index'],support_sha256=q['support_sha256']) for q in eligible],
                   selection='All annotated answer-bearing turns occur verbatim after whitespace normalization in the prefix; no answer-based filtering.',
                   limitation='Other unannotated turns may lie outside prefix; structural acceptance is not semantic memory correctness.')
    return chunks,eligible,state,excluded,receipt


def tensor_bytes(tensors):
    storages = {}
    for t in tensors:
        if t is not None:
            storage=t.untyped_storage()
            storages[(str(t.device),storage.data_ptr())]=storage.nbytes()
    return sum(storages.values())


def cache_bytes(cache):
    if cache is None:
        return 0
    return tensor_bytes(t for layer in cache.layers for t in (layer.keys,layer.values))


def run(args, out, manifest):
    source=ROOT/'outputs/phase4/mem0_8b_guarded'
    os.environ.update(OPENAI_API_KEY='EMPTY',OPENAI_BASE_URL='http://127.0.0.1:8030/v1',
                      HF_HUB_OFFLINE='1',HF_HOME=str(ROOT/'hf'),MEM0_TELEMETRY='False',
                      MEM0_DIR=str(out/'mem0_home'),NLTK_DATA=str(ROOT/'nltk_data'),
                      TIKTOKEN_CACHE_DIR=str(ROOT/'tiktoken-cache'),TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for key in ['AZURE_OPENAI_API_KEY','AZURE_OPENAI_ENDPOINT','QWEN_EMBED_BACKEND']:
        os.environ.pop(key,None)
    sys.path.insert(0,str(ROOT/'external/agent-memory-profiling'))
    import nltk
    def offline_nltk(name,*a,**kw):
        kind='tokenizers' if name in ('punkt','punkt_tab') else 'corpora'
        assert (ROOT/'nltk_data'/kind/name).exists()
        return True
    nltk.download=offline_nltk
    from conversation_creator import ConversationCreator
    from agent import AgentWrapper
    import torch
    torch.set_num_threads(1)
    old=json.loads((source/'both_manifest.json').read_text())
    creator=ConversationCreator(old['agent_config'],old['dataset_config'])
    chunks,questions,expected,excluded,receipt=audit(source,creator)
    manifest['audit']=receipt
    write(out/'manifest.json',manifest)
    if args.audit_only:
        return
    shutil.copytree(source/'state',out/'state')
    # Remove the failed operation from the COPY's history and index only.
    connection=sqlite3.connect(out/'state/mem0/history.db')
    connection.execute('delete from history where id=?',(excluded[0]['id'],))
    connection.commit();connection.close()
    cfg=dict(old['agent_config'],output_dir=str(out))
    agent=AgentWrapper(cfg,old['dataset_config'],str(out/'state'))
    agent.memory.vector_store.delete(excluded[0]['memory_id'])
    user_id='context_0_'+old['dataset_config']['sub_dataset']
    records=agent.memory.get_all(user_id=user_id,limit=10000)['results']
    assert {x['id']:x['memory'] for x in records}==expected
    manifest['audit']['restored_index_equals_history']=True
    from methods.embedding_retriever import TextRetriever
    retriever=TextRetriever(embedding_model_name='Qwen/Qwen3-Embedding-0.6B',
                           openai_base_url='http://127.0.0.1:8030/v1',openai_api_key='EMPTY',openai_embedding_dim=1024)
    start=time.perf_counter()
    retriever.build_vectorstore(chunks)
    manifest['raw_index_build_seconds']=time.perf_counter()-start
    requests=[]
    for pair,q in enumerate(questions):
        raw=retriever.retrieve(q['question'],top_k=5)
        facts=agent.memory.search(query=q['question'],user_id=user_id,limit=10)['results']
        contexts={'raw':'\n\n'.join(raw),'mem0':'\n'.join('- '+x['memory'] for x in facts)}
        for strategy in (('raw','mem0') if pair%2==0 else ('mem0','raw')):
            messages=[dict(role='system',content='Answer the question using only the supplied memory. '
                           'Be concise. If the information is insufficient, say so.\n\n'+contexts[strategy]),
                      dict(role='user',content=q['question']+'\n\nQuestion reference date: '+str(q['date']))]
            requests.append(dict(index=q['index'],strategy=strategy,messages=messages,answer=q['answer']))
    write(out/'requests_private.json',requests)
    agent.memory.vector_store.client.close()
    manifest['request_count']=len(requests)
    manifest['request_sha256']=sha(out/'requests_private.json')
    write(out/'manifest.json',manifest)
    print('Prepared',len(requests),'paired requests; loading 32B on GPU 1',flush=True)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from utils.eval_other_utils import post_process
    models=json.loads((ROOT/'models.json').read_text())
    path=models['Qwen/Qwen3-32B']['path']
    device='cuda:1'
    torch.cuda.set_device(device)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,device_map=device,
                                             attn_implementation='sdpa',local_files_only=True).eval()
    weights=tensor_bytes(model.parameters())
    model_buffers=tensor_bytes(model.buffers())
    import pynvml
    pynvml.nvmlInit()
    nvml_device=pynvml.nvmlDeviceGetHandleByIndex(1)
    manifest.update(torch_version=torch.__version__,transformers_version=__import__('transformers').__version__,
                    gpu=torch.cuda.get_device_name(),weight_bytes=weights,model_buffer_bytes=model_buffers,
                    model_revision=models['Qwen/Qwen3-32B']['revision'],backend='HF eager SDPA, BF16, greedy, thinking disabled',
                    scope='Sequential B=1 offline query, 8B not resident; synchronized instrumentation is not optimized serving latency.')
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    summary=[]
    def sample(cache,phase,step,interval_peak=None):
        torch.cuda.synchronize(device)
        allocated=torch.cuda.memory_allocated(device)
        reserved=torch.cuda.memory_reserved(device)
        kv=cache_bytes(cache)
        assert allocated >= weights+kv
        process_memory=next((p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(nvml_device)
                             if p.pid==os.getpid()),None)
        return dict(time=time.time(),phase=phase,step=step,weights_bytes=weights,kv_bytes=kv,
                    allocated_bytes=allocated,reserved_bytes=reserved,
                    other_live_allocated_bytes=allocated-weights-kv,
                    allocator_slack_bytes=reserved-allocated,interval_peak_allocated_bytes=interval_peak,
                    nvml_process_used_bytes=process_memory,
                    nvml_outside_reserved_estimate=None if process_memory is None else process_memory-reserved)
    manifest['loaded_sample']=sample(None,'model_loaded',-1)
    write(out/'manifest.json',manifest)
    with torch.inference_mode():
        for req in requests:
            gc.collect();torch.cuda.empty_cache()
            ids=tokenizer.apply_chat_template(req['messages'],tokenize=True,add_generation_prompt=True,
                                              enable_thinking=False,return_tensors='pt').to(device)
            length=ids.shape[1]
            assert length+args.max_tokens <= 32768
            generated=[];cache=None;samples=[];elapsed=[]
            start=time.perf_counter()
            for step in range(args.max_tokens):
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                tick=time.perf_counter()
                output=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=output.past_key_values
                token=int(output.logits[0,-1].argmax().item())
                torch.cuda.synchronize(device)
                elapsed.append(time.perf_counter()-tick)
                samples.append(sample(cache,'prefill' if step==0 else 'decode',step,
                                      torch.cuda.max_memory_allocated(device)))
                generated.append(token)
                del output
                if token in eos:
                    break
                ids=torch.tensor([[token]],device=device,dtype=torch.long)
            text=tokenizer.decode(generated,skip_special_tokens=True)
            metrics,_=post_process({'output':text},req['answer'],old['dataset_config'])
            key=f"{req['index']:02d}_{req['strategy']}"
            write(out/(key+'_private.json'),dict(**req,output=text,output_ids=generated,samples=samples))
            measured=dict(index=req['index'],strategy=req['strategy'],input_tokens=length,output_tokens=len(generated),
                          stopped_eos=generated[-1] in eos,capped=generated[-1] not in eos,
                          forward_seconds=sum(elapsed),prefill_forward_seconds=elapsed[0],wall_seconds=time.perf_counter()-start,
                          final_live=samples[-1],prefill_live=samples[0],
                          peak_interval_allocated_bytes=max(s['interval_peak_allocated_bytes'] for s in samples),
                          metrics=metrics,private_receipt_sha256=sha(out/(key+'_private.json')))
            assert samples[-1]['kv_bytes']==(length+len(generated)-1)*262144
            summary.append(measured)
            write(out/'summary.json',dict(status='running',results=summary,manifest_sha256=sha(out/'manifest.json')))
            print(key,'input',length,'output',len(generated),'kv',samples[-1]['kv_bytes'],'metrics',metrics,flush=True)
            del ids,cache,samples
    for relative,digest in receipt['source_files'].items():
        assert sha(source/relative)==digest
    manifest.update(status='complete',end=time.time(),original_state_unchanged=True)
    write(out/'manifest.json',manifest)
    write(out/'summary.json',dict(status='complete',results=summary,manifest_sha256=sha(out/'manifest.json')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='finite_prefix_01')
    parser.add_argument('--max-tokens',type=int,default=256)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    assert args.run.replace('_','').isalnum()
    out=ROOT/'outputs'/args.run
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise RuntimeError('Choose a fresh run; evidence is immutable')
    manifest=dict(status='running',start=time.time(),pid=os.getpid(),args=vars(args),runner_sha256=sha(__file__))
    write(out/'manifest.json',manifest)
    try:
        run(args,out,manifest)
        if args.audit_only:
            manifest.update(status='audit_complete',end=time.time());write(out/'manifest.json',manifest)
    except Exception:
        manifest.update(status='failed',error=traceback.format_exc(),end=time.time())
        write(out/'manifest.json',manifest)
        raise


if __name__=='__main__':
    main()
