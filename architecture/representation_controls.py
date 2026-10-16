"""Bounded remote controls: seven questions, 512-token raw retrieval, two audits."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from architecture.finite_prefix import ROOT, audit, sha, write, tensor_bytes, cache_bytes


def run(out, manifest):
    os.environ.update(OPENAI_API_KEY='EMPTY',OPENAI_BASE_URL='http://127.0.0.1:8030/v1',HF_HUB_OFFLINE='1',
        HF_HOME=str(ROOT/'hf'),MEM0_TELEMETRY='False',NLTK_DATA=str(ROOT/'nltk_data'),
        TIKTOKEN_CACHE_DIR=str(ROOT/'tiktoken-cache'),TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,str(ROOT/'external/agent-memory-profiling'))
    import nltk
    nltk.download=lambda name,*a,**kw: (ROOT/'nltk_data'/('tokenizers' if name in ('punkt','punkt_tab') else 'corpora')/name).exists()
    from conversation_creator import ConversationCreator
    from methods.embedding_retriever import TextRetriever
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from utils.eval_other_utils import post_process
    import torch
    torch.set_num_threads(1)
    original=ROOT/'outputs/finite_prefix_01'
    source=ROOT/'outputs/phase4/mem0_8b_guarded'
    old=json.loads((source/'both_manifest.json').read_text())
    creator=ConversationCreator(old['agent_config'],old['dataset_config'])
    chunks,questions,state,_,receipt=audit(source,creator)
    assert [q['index'] for q in questions]==[3,28,30,31,39,48,54]
    models=json.loads((ROOT/'models.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(models['Qwen/Qwen3-32B']['path'],local_files_only=True)
    documents=[];locations=[]
    for parent,chunk in enumerate(chunks):
        tokens=tokenizer.encode(chunk,add_special_tokens=False)
        for start in range(0,len(tokens),448):
            end=min(start+512,len(tokens))
            documents.append(tokenizer.decode(tokens[start:end],skip_special_tokens=False))
            locations.append(dict(parent=parent,start_token=start,end_token=end))
            if end==len(tokens): break
    write(out/'documents_private.json',dict(documents=documents,locations=locations))
    manifest.update(audit=receipt,chunk_tokens=512,overlap_tokens=64,top_k=5,document_count=len(documents),
                    tokenizer_revision=models['Qwen/Qwen3-32B']['revision'],
                    original_manifest_sha256=sha(original/'manifest.json'))
    retriever=TextRetriever(embedding_model_name='Qwen/Qwen3-Embedding-0.6B',openai_base_url='http://127.0.0.1:8030/v1',
                           openai_api_key='EMPTY',openai_embedding_dim=1024)
    start=time.perf_counter();retriever.build_vectorstore(documents)
    manifest['index_build_seconds']=time.perf_counter()-start
    manifest['documents_sha256']=sha(out/'documents_private.json')
    requests=[]
    for q in questions:
        context='\n\n'.join(retriever.retrieve(q['question'],top_k=5))
        # Preserve the original controlled prompt and reference date verbatim.
        prior=json.loads((original/f"{q['index']:02d}_raw_private.json").read_text())
        header=prior['messages'][0]['content'].split('\n\n',1)[0]
        messages=[dict(role='system',content=header+'\n\n'+context),prior['messages'][1]]
        requests.append(dict(index=q['index'],strategy='raw_512',messages=messages,answer=q['answer']))
    write(out/'requests_private.json',requests)
    # Bounded audit exports only hashes/counts; manual interpretation is in report.
    audits=[]
    needles={30:['chandelier','aunt'],39:['social','break','january','february']}
    for i in (30,39):
        raw=json.loads((original/f'{i:02d}_raw_private.json').read_text())
        mem=json.loads((original/f'{i:02d}_mem0_private.json').read_text())
        matches=[dict(id=k,text=v) for k,v in state.items() if any(w in v.lower() for w in needles[i])]
        private=dict(index=i,state_matches=matches,question=raw['messages'][1]['content'],answer=raw['answer'],
                     raw_output=raw['output'],mem0_output=mem['output'],mem0_context=mem['messages'][0]['content'])
        write(out/f'{i}_audit_private.json',private)
        audits.append(dict(index=i,checked_state_records=len(state),keyword_candidate_count=len(matches),
                           keyword_screen=needles[i],private_sha256=sha(out/f'{i}_audit_private.json'),
                           original_raw_sha256=sha(original/f'{i:02d}_raw_private.json'),
                           original_mem0_sha256=sha(original/f'{i:02d}_mem0_private.json')))
    manifest['two_case_audit']=audits
    manifest['request_sha256']=sha(out/'requests_private.json')
    write(out/'manifest.json',manifest)
    print('Prepared',len(documents),'raw fragments and',len(requests),'queries',flush=True)
    device='cuda:1';torch.cuda.set_device(device)
    model=AutoModelForCausalLM.from_pretrained(models['Qwen/Qwen3-32B']['path'],torch_dtype=torch.bfloat16,
        device_map=device,attn_implementation='sdpa',local_files_only=True).eval()
    weights=tensor_bytes(model.parameters());eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    manifest.update(weight_bytes=weights,model_revision=models['Qwen/Qwen3-32B']['revision'],
                    torch_version=torch.__version__,transformers_version=__import__('transformers').__version__,
                    backend='HF eager SDPA, BF16, greedy, thinking disabled; sequential instrumentation')
    results=[]
    with torch.inference_mode():
        for req in requests:
            gc.collect();torch.cuda.empty_cache()
            ids=tokenizer.apply_chat_template(req['messages'],tokenize=True,add_generation_prompt=True,
                enable_thinking=False,return_tensors='pt').to(device)
            length=ids.shape[1];generated=[];cache=None;samples=[];elapsed=[]
            for step in range(256):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                response=model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=response.past_key_values;token=int(response.logits[0,-1].argmax().item())
                torch.cuda.synchronize();elapsed.append(time.perf_counter()-tick)
                allocated=torch.cuda.memory_allocated();kv=cache_bytes(cache)
                samples.append(dict(step=step,kv_bytes=kv,allocated_bytes=allocated,
                    other_live_bytes=allocated-weights-kv,reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                generated.append(token);del response
                if token in eos: break
                ids=torch.tensor([[token]],device=device,dtype=torch.long)
            text=tokenizer.decode(generated,skip_special_tokens=True)
            metrics,_=post_process({'output':text},req['answer'],old['dataset_config'])
            private=out/f"{req['index']:02d}_private.json"
            write(private,dict(**req,output=text,output_ids=generated,samples=samples))
            assert kv==(length+len(generated)-1)*262144
            results.append(dict(index=req['index'],strategy='raw_512',input_tokens=length,output_tokens=len(generated),
                capped=token not in eos,metrics=metrics,forward_seconds=sum(elapsed),prefill_forward_seconds=elapsed[0],
                final_live=samples[-1],peak_allocated_bytes=max(s['peak_allocated_bytes'] for s in samples),
                private_sha256=sha(private)))
            write(out/'summary.json',dict(status='running',results=results))
            print(req['index'],length,len(generated),metrics,flush=True)
            del ids,cache,samples
    for path,digest in receipt['source_files'].items(): assert sha(source/path)==digest
    manifest.update(status='complete',end=time.time(),original_state_unchanged=True)
    write(out/'manifest.json',manifest)
    write(out/'summary.json',dict(status='complete',results=results,manifest_sha256=sha(out/'manifest.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='representation_01');args=p.parse_args()
    assert args.run.replace('_','').isalnum()
    out=ROOT/'outputs'/args.run;out.mkdir(exist_ok=True,parents=True)
    assert not (out/'manifest.json').exists()
    m=dict(status='running',start=time.time(),pid=os.getpid(),runner_sha256=sha(__file__))
    write(out/'manifest.json',m)
    try: run(out,m)
    except Exception:
        m.update(status='failed',error=traceback.format_exc());write(out/'manifest.json',m);raise


if __name__=='__main__': main()
