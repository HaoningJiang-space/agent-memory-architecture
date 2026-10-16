"""Execute pinned upstream algorithms; keep all raw traces and states remote."""
import argparse
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', choices=['mem0', 'embedrag', 'graphrag', 'bm25'], required=True)
parser.add_argument('--run', required=True)
parser.add_argument('--stage', choices=['construction', 'query', 'both'], default='both')
parser.add_argument('--dataset', default='longmemeval.json')
parser.add_argument('--max-chunks', type=int, default=0, help='Diagnostic subset only; 0 means all')
parser.add_argument('--max-queries', type=int, default=0)
parser.add_argument('--answer-max-tokens', type=int, default=256)
parser.add_argument('--construction-model', choices=['Qwen/Qwen3-8B', 'Qwen/Qwen3-32B'], default='Qwen/Qwen3-8B')
parser.add_argument('--wait-for-qa', action='store_true', help='Keep native in-memory index alive until QA server is ready')
parser.add_argument('--precompute-ner', action='store_true', help='Software control: batch spaCy before concurrent concept LLM calls')
parser.add_argument('--trace-file', default='outputs/api_trace.jsonl')
parser.add_argument('--update-control', choices=['native', 'flat_snapshot', 'flat_delta'], default='native')
args = parser.parse_args()
if args.update_control != 'native' and args.algorithm != 'mem0':
    parser.error('--update-control is only available for Mem0')
router_health = json.load(urllib.request.urlopen('http://127.0.0.1:8030/health', timeout=5))
if router_health.get('trace_path') != str((ROOT/args.trace_file).resolve()):
    raise RuntimeError('Router trace destination does not match --trace-file')
out = ROOT/'outputs'/args.run
out.mkdir(parents=True, exist_ok=True)
if (out/f'{args.stage}_manifest.json').exists():
    raise RuntimeError('Existing run manifest: choose a new run name')
(out/f'{args.stage}_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
os.environ.update(OPENAI_API_KEY='EMPTY', OPENAI_BASE_URL='http://127.0.0.1:8030/v1',
                  HF_HUB_OFFLINE='1', HF_HOME=str(ROOT/'hf'), MEM0_TELEMETRY='False',
                  MEM0_DIR=str(out/'mem0_home'), NLTK_DATA=str(ROOT/'nltk_data'),
                  TIKTOKEN_CACHE_DIR=str(ROOT/'tiktoken-cache'), VLLM_GUIDED_DECODING='1',
                  MAB_RAG_CACHE_DIR=str(out/'rag_cache'), TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1')
for key in ['OPENROUTER_API_KEY', 'AZURE_OPENAI_API_KEY', 'QWEN_EMBED_BACKEND']:
    os.environ.pop(key, None)
sys.path.insert(0, str(ROOT/'external/agent-memory-profiling'))
import nltk
def offline_nltk(name, *a, **kw):
    kind = 'tokenizers' if name in ['punkt', 'punkt_tab'] else 'corpora'
    assert (ROOT/'nltk_data'/kind/name).exists(), f'Missing mirrored NLTK resource {name}'
    return True
nltk.download = offline_nltk
if args.algorithm == 'graphrag':
    import spacy
    original_load = spacy.load
    def local_spacy(name, **kw):
        if name == 'en_core_web_sm':
            name = json.loads((ROOT/'spacy_model.json').read_text())['path']
        return original_load(name, **kw)
    spacy.load = local_spacy

from agent import AgentWrapper
from conversation_creator import ConversationCreator
from profiler.memory_tracker import ScopedTracker
import torch
torch.set_num_threads(1)

def write(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str)+'\n')
    tmp.replace(path)

def append(name, obj):
    with (out/name).open('a') as f:
        f.write(json.dumps(obj, default=str)+'\n')

names = {'mem0':'Structure_rag_mem0', 'embedrag':'Embedding_rag_qwen3_embedding_0_6b',
         'graphrag':'Structure_rag_graph_rag', 'bm25':'rag_bm25'}
base = 'http://127.0.0.1:8030/v1'
cfg = dict(agent_name=names[args.algorithm], model='Qwen/Qwen3-32B',
           memory_construction_llm=args.construction_model, qa_generation_llm='Qwen/Qwen3-32B',
           openai_base_url=base, qa_openai_base_url=base, api_key='EMPTY',
           temperature=0.0, input_length_limit=32000, buffer_length=1000,
           retrieve_num=10 if args.algorithm in ['mem0','graphrag'] else 5, output_dir=str(out), embedding_model='Qwen/Qwen3-Embedding-0.6B',
           embedding_base_url=base, embedding_api_key_env='OPENAI_API_KEY', embedding_dim=1024,
           graph_rag_max_tokens=1024, graph_rag_llm_context_window=32000)
if args.algorithm == 'mem0':
    cfg['agent_chunk_size'] = 4096
    cfg['mem0_llm'] = dict(provider='openai', config=dict(model=args.construction_model,
                           temperature=0.0, max_tokens=4096, openai_base_url=base, api_key='EMPTY'))
data_cfg = dict(dataset='Accurate_Retrieval', sub_dataset='longmemeval_s*',
                context_max_length=400000, max_test_samples=1, chunk_size=4096,
                generation_max_length=args.answer_max_tokens, local_json=str(ROOT/args.dataset), random_seed=42, seed=42)
manifest = dict(args=vars(args), agent_config=cfg, dataset_config=data_cfg,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dataset_sha256=hashlib.sha256((ROOT/args.dataset).read_bytes()).hexdigest(),
                source_commit='30a157b7b49f732ee4b961cb19d7f877cf475f81',
                cpu_runtime=dict(OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,torch_num_threads=1),
                models=json.loads((ROOT/'models.json').read_text()), start=time.time(),
                status='running', validity='pending')
if args.update_control != 'native':
    source = ROOT/'protocol_controls.py'
    (out/'protocol_controls_snapshot.py').write_bytes(source.read_bytes())
    manifest['software_control'] = dict(name=args.update_control,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Prompt/schema and pre-application reference validation; bounded retries. '
              'This is a modified protocol, not unmodified native Mem0 or proof of semantic quality.')
write(out/f'{args.stage}_manifest.json', manifest)

class ErrorCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []
    def emit(self, record):
        self.messages.append(record.getMessage())
errors = ErrorCapture()
logging.getLogger().addHandler(errors)
class CheckedStdout:
    def write(self, value):
        if 'Error extracting concepts:' in value:
            errors.messages.append(value)
        return sys.__stdout__.write(value)
    def flush(self):
        sys.__stdout__.flush()
sys.stdout = CheckedStdout()
tracker = ScopedTracker()
if args.algorithm == 'graphrag' and args.precompute_ner:
    from methods.graph_rag import KnowledgeGraph
    native_extract = KnowledgeGraph._extract_concepts
    def batched_ner_extract(graph, splits, llm):
        begin = time.time()
        texts = [s.page_content for s in splits]
        nlp = graph.nlp
        docs = list(nlp.pipe(texts, batch_size=32, n_process=1))
        def entities(doc):
            return [(e.start_char,e.end_char,e.label_,e.text) for e in doc.ents]
        # A bounded semantic parity control; all documents still run through
        # the full native spaCy pipeline. No pipeline component is disabled.
        for i in range(min(16,len(docs))):
            assert entities(nlp(texts[i])) == entities(docs[i]), 'NER batch parity failed'
        by_text = dict(zip(texts,docs))
        append('ner_batch.jsonl',dict(start=begin,end=time.time(),documents=len(docs),
                                     parity_checked=min(16,len(docs)),parity_passed=True))
        graph.nlp = lambda text: by_text[text]
        try:
            return native_extract(graph,splits,llm)
        finally:
            graph.nlp = nlp
    KnowledgeGraph._extract_concepts = batched_ner_extract
monitor = subprocess.Popen([sys.executable, str(ROOT/'watch_gpu.py'), str(out/f'{args.stage}_gpu.jsonl')],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def mark(phase, index):
    context = dict(run=args.run, algorithm=args.algorithm, phase=phase, index=index)
    tracker.current_context = context
    write(ROOT/'run_context.json', context)
    return context

update_control = None


def measured(fn, phase, index):
    context = mark(phase, index)
    begin = time.time()
    count = len(errors.messages)
    calls = len(tracker.call_log)
    trace_path = ROOT/args.trace_file
    offset = trace_path.stat().st_size if trace_path.exists() else 0
    value = None
    validation, failure = 'failed', None
    try:
        value = fn()
        if len(errors.messages) > count:
            raise RuntimeError('Upstream logged errors: '+repr(errors.messages[count:]))
        if trace_path.exists():
            with trace_path.open() as f:
                f.seek(offset)
                for line in f:
                    row = json.loads(line)
                    if row.get('context') != context:
                        continue
                    if update_control and row['id'] in update_control.handled_trace_ids:
                        # The validator checked the accepted response, and accounted for
                        # failed attempts before returning anything to the state writer.
                        continue
                    reasons = [row.get('finish_reason')]
                    reasons += [c.get('finish_reason') for c in row.get('response', {}).get('choices', [])]
                    if 'length' in reasons:
                        raise RuntimeError('Generation truncated at token cap; run is invalid')
                    if row.get('error') or row.get('status', 200) != 200:
                        raise RuntimeError('Inference request failed; inspect raw API trace')
        validation = 'passed'
        return value
    except BaseException as exc:
        failure = repr(exc)
        manifest['failed_context'] = context
        raise
    finally:
        append('steps.jsonl', dict(**context, start=begin, end=time.time(), result=value,
                                  validation=validation, error=failure))
        for row in tracker.call_log[calls:]:
            append('calls.jsonl', row)

try:
    creator = ConversationCreator(cfg, data_cfg)
    chunks, questions = creator.get_chunks()[0], creator.get_query_and_answers()[0]
    manifest.update(total_chunks=len(chunks), total_queries=len(questions),
                    context_chars=len(creator.contexts[0]))
    write(out/f'{args.stage}_manifest.json', manifest)
    if args.stage == 'query':
        assert args.algorithm == 'mem0', 'Other native agents lack state reload; use stage both'
        prior = json.loads((out/'construction_manifest.json').read_text())
        assert prior['status'] == 'complete' and prior['validity'] == 'valid'
        assert prior['args']['construction_model'] == args.construction_model
        assert prior['args'].get('update_control','native') == args.update_control
    elif (out/'state').exists():
        raise RuntimeError('Existing state: choose a new run name to avoid duplicate ingestion')
    with tracker:
        mark('initialization', -1)
        agent = AgentWrapper(cfg, data_cfg, str(out/'state'), force_rerun=False)
        if args.algorithm == 'mem0':
            if args.update_control != 'native':
                from protocol_controls import UpdateControl
                update_control = UpdateControl(agent.memory.llm.generate_response, args.update_control,
                    lambda row: append('protocol_validation.jsonl',
                                       dict(row, context=dict(tracker.current_context))), ROOT/args.trace_file)
                agent.memory.llm.generate_response = update_control
            # Observe real index operations without changing order or issuing extra queries.
            for name in ['search', 'insert', 'update', 'delete']:
                original = getattr(agent.memory.vector_store, name)
                def traced(*a, _fn=original, _name=name, **kw):
                    start = time.time()
                    value = _fn(*a, **kw)
                    append('index.jsonl', dict(operation=_name, start=start, end=time.time(),
                           context=dict(tracker.current_context),
                           result_count=len(value) if hasattr(value, '__len__') else None))
                    return value
                setattr(agent.memory.vector_store, name, traced)
        if args.stage in ['construction', 'both']:
            for i, chunk in enumerate(chunks[:args.max_chunks or None]):
                measured(lambda c=chunk: agent.send_message(c, memorizing=True, context_id=0), 'construction', i)
                print(json.dumps(dict(progress='chunk', completed=i+1, total=len(chunks))), flush=True)
            measured(lambda: agent.finalize_memory_construction(context_id=0), 'index_finalize', -1)
            agent.save_agent()
        if args.stage in ['query', 'both']:
            if args.wait_for_qa:
                deadline = time.monotonic()+7200
                while True:
                    try:
                        urllib.request.urlopen('http://127.0.0.1:8032/health', timeout=3).read()
                        break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic()>deadline:
                            raise RuntimeError('QA server readiness deadline exceeded')
                        manifest['waiting_for_qa'] = True
                        write(out/f'{args.stage}_manifest.json', manifest)
                        time.sleep(10)
                manifest['waiting_for_qa'] = False
            for i, (query, answer, qid) in enumerate(questions[:args.max_queries or None]):
                response = measured(lambda: agent.send_message(query, memorizing=False, query_id=i, context_id=0), 'query', i)
                append('answers.jsonl', dict(index=i, qa_id=qid, query=query, answer=answer, response=response))
                print(json.dumps(dict(progress='query', completed=i+1, total=len(questions))), flush=True)
    manifest.update(status='complete', validity='diagnostic_subset' if args.max_chunks or args.max_queries or 'tiny' in args.dataset else 'valid')
except BaseException as exc:
    manifest.update(status='failed', validity='invalid', error=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    monitor.terminate()
    monitor.wait(timeout=10)
    manifest.update(end=time.time(), logged_errors=errors.messages,
                    totals={k:getattr(tracker,k) for k in ['chat_calls','embedding_calls','prompt_tokens','completion_tokens','total_llm_latency','retrieval_latency','generation_latency']})
    write(out/f'{args.stage}_manifest.json', manifest)
