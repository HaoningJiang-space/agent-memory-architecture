"""Remote bounded native SimpleMem/LightMem runs; all raw content stays remote.

Deployment adapters use the existing local Qwen services. Native prompts,
memory representations and retrieval implementations are preserved. This is
shape/dependency coverage, not a paper reproduction or a quality comparison.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
LIGHT_REV = '8449d574df6bae1bdf3314a1564da65e2f37e046'
MODEL = 'Qwen/Qwen3-8B'
MAIN = 'Qwen/Qwen3-32B'
EMBED = 'Qwen/Qwen3-Embedding-0.6B'
BASE_URL = 'http://127.0.0.1:8030/v1'
PHASE = 'initialization'
LOCK = threading.Lock()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def append(path, value):
    with LOCK, path.open('a') as f:
        f.write(json.dumps(value, default=str) + '\n')


def set_phase(algorithm, phase):
    global PHASE
    PHASE = phase
    write(ROOT / 'run_context.json', dict(run='family_01', algorithm=algorithm, phase=phase))


def install_call_audit(out):
    from openai.resources.chat.completions import Completions
    original = Completions.create

    def create(self, *args, **kwargs):
        assert str(self._client.base_url).startswith(BASE_URL)
        assert kwargs['model'] in (MODEL, MAIN)
        assert not kwargs.get('stream', False)
        kwargs['temperature'] = 0
        kwargs.setdefault('max_tokens', 8192 if kwargs['model'] == MODEL else 512)
        row = dict(id=str(uuid.uuid4()), phase=PHASE, start=time.time(), thread=threading.get_ident(),
                   request=kwargs, validated=False)
        try:
            response = original(self, *args, **kwargs)
            row.update(response=response.model_dump(), end=time.time())
            assert response.choices[0].finish_reason == 'stop', 'Capped/failed generation is not valid evidence'
            assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0
            row['validated'] = True
            return response
        except Exception as exc:
            row.update(end=time.time(), error=str(exc))
            raise
        finally:
            append(out / 'llm_private.jsonl', row)
    Completions.create = create


def instrument(obj, names, out):
    for name in names:
        if not hasattr(obj, name):
            continue
        original = getattr(obj, name)
        def wrapped(*args, _name=name, _original=original, **kwargs):
            row = dict(operation=_name, phase=PHASE, start=time.time(), thread=threading.get_ident())
            try:
                result = _original(*args, **kwargs)
                row.update(end=time.time(), status='ok', result_count=len(result) if isinstance(result, (list, dict)) else None)
                return result
            except Exception as exc:
                row.update(end=time.time(), status='error', error=str(exc))
                raise
            finally:
                append(out / 'operations.jsonl', row)
        setattr(obj, name, wrapped)


def source_history():
    data = json.loads((ROOT / 'longmemeval.json').read_text())[0]
    context = ast.literal_eval(data['context'])
    selected = []
    for position in range(0, len(context), 2):
        stamp = context[position].removeprefix('Chat Time: ')
        for message in context[position + 1]:
            selected.append(dict(message, time_stamp=stamp))
            if len(selected) == 48:
                return selected, data
    raise ValueError('Insufficient source history')


def simplemem(out, history, data, manifest):
    path = ROOT / 'external/agent-memory-profiling/methods/SimpleMem'
    sys.path.insert(0, str(path))
    import numpy as np
    from openai import OpenAI
    import config
    config.WINDOW_SIZE = 16
    config.OVERLAP_SIZE = 2
    import main as native
    from models.memory_entry import Dialogue
    from utils.llm_client import LLMClient

    class LocalEmbedding:
        dimension = 1024
        def __init__(self):
            self.client = OpenAI(api_key='EMPTY', base_url=BASE_URL)
        def encode(self, texts, is_query=False):
            if isinstance(texts, str): texts = [texts]
            response = self.client.embeddings.create(model=EMBED, input=texts, encoding_format='float')
            return np.asarray([r.embedding for r in response.data], dtype=np.float32)
        def encode_single(self, text, is_query=False): return self.encode([text], is_query)[0]
        def encode_documents(self, texts): return self.encode(texts)
        def encode_query(self, texts): return self.encode(texts, True)
    native.EmbeddingModel = LocalEmbedding
    system = native.SimpleMemSystem(api_key='EMPTY', model=MODEL, base_url=BASE_URL,
        db_path=str(out / 'state'), table_name='family', use_streaming=False, enable_thinking=False,
        enable_parallel_processing=True, max_parallel_workers=4, enable_planning=True,
        enable_reflection=True, max_reflection_rounds=2, enable_parallel_retrieval=True, max_retrieval_workers=3)
    system.answer_generator.llm_client = LLMClient(api_key='EMPTY', model=MAIN, base_url=BASE_URL,
                                                   use_streaming=False, enable_thinking=False)
    instrument(system.vector_store, ['add_entries', 'semantic_search', 'keyword_search', 'structured_search'], out)
    dialogues = [Dialogue(dialogue_id=i + 1, speaker=m['role'], content=m['content'],
                    timestamp=datetime.strptime(m['time_stamp'], '%Y/%m/%d (%a) %H:%M').isoformat())
                 for i, m in enumerate(history)]
    set_phase('simplemem', 'construction')
    system.add_dialogues(dialogues)
    system.finalize()
    entries = system.get_all_memories()
    assert entries and system.vector_store._fts_initialized
    write(out / 'state_private.json', [e.model_dump() for e in entries])
    manifest.update(records=len(entries), window_size=16, overlap=2, build_workers=4,
                    native_revision='agent-memory-profiling@30a157b7b49f732ee4b961cb19d7f877cf475f81')
    manifest['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in
        [path / 'core/memory_builder.py', path / 'core/hybrid_retriever.py', path / 'database/vector_store.py']}
    for i, question in enumerate(data['questions'][:3]):
        set_phase('simplemem', f'query_{i}')
        contexts = system.hybrid_retriever.retrieve(question)
        answer = system.answer_generator.generate_answer(question, contexts)
        write(out / f'query_{i}_private.json', dict(question=question, answer=answer,
                                                  contexts=[e.model_dump() for e in contexts]))


def lightmem(out, history, data, manifest):
    path = ROOT / f'external/LightMem-{LIGHT_REV}'
    sys.path.insert(0, str(path / 'src'))
    from lightmem.memory.lightmem import LightMemory
    from openai import OpenAI
    cfg = dict(pre_compress=False, topic_segment=False, messages_use='hybrid', metadata_generate=True,
        text_summary=True, extract_threshold=.1, update='offline', extraction_mode='flat',
        memory_manager=dict(model_name='openai', configs=dict(model=MODEL, api_key='EMPTY',
            openai_base_url=BASE_URL, max_tokens=8192, temperature=0, top_p=1)),
        index_strategy='embedding', retrieve_strategy='embedding',
        text_embedder=dict(model_name='openai', configs=dict(model=EMBED, api_key='EMPTY',
            openai_base_url=BASE_URL, embedding_dims=1024)),
        embedding_retriever=dict(model_name='qdrant', configs=dict(collection_name='family',
            embedding_model_dims=1024, path=str(out / 'state'), on_disk=True)))
    system = LightMemory.from_config(cfg)
    instrument(system.embedding_retriever, ['insert', 'search', 'get_all', 'update', 'delete'], out)
    set_phase('lightmem', 'construction')
    for i in range(0, len(history), 2):
        result = system.add_memory(history[i:i + 2], force_extract=i + 2 == len(history))
        append(out / 'construction_private.jsonl', dict(index=i, result=result))
    before = system.embedding_retriever.get_all()
    assert before
    write(out / 'before_maintenance_private.json', before)
    set_phase('lightmem', 'maintenance')
    system.construct_update_queue_all_entries(top_k=20, keep_top_n=10, max_workers=4)
    system.offline_update_all_entries(score_threshold=.9, max_workers=5)
    after = system.embedding_retriever.get_all()
    write(out / 'state_private.json', after)
    manifest.update(records_before_maintenance=len(before), records=len(after), config=cfg,
                    native_revision=LIGHT_REV, native_tokens=system.get_token_statistics())
    manifest['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in
        [path / 'src/lightmem/memory/lightmem.py', path / 'src/lightmem/factory/memory_manager/openai.py',
         path / 'src/lightmem/factory/memory_buffer/short_term_memory.py']}
    client = OpenAI(api_key='EMPTY', base_url=BASE_URL)
    for i, question in enumerate(data['questions'][:3]):
        set_phase('lightmem', f'query_{i}')
        memories = system.retrieve(question, limit=20)
        response = client.chat.completions.create(model=MAIN, messages=[dict(role='system',
            content='Answer using the supplied memory. Be concise. If information is insufficient, say so.'),
            dict(role='user', content=question + '\nQuestion reference date: ' + str(data['metadata']['question_dates'][i]) +
                 '\nMemories:\n' + '\n'.join(memories))])
        write(out / f'query_{i}_private.json', dict(question=question, memories=memories, response=response.model_dump()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', choices=['simplemem', 'lightmem'])
    parser.add_argument('--session', default='family_01')
    args = parser.parse_args()
    assert args.session.replace('_', '').isalnum()
    out = ROOT / 'outputs' / args.session / args.algorithm
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENAI_API_KEY='EMPTY', OPENAI_BASE_URL=BASE_URL, HF_HUB_OFFLINE='1',
        HF_HOME=str(ROOT/'hf'), TIKTOKEN_CACHE_DIR=str(ROOT/'tiktoken-cache'), TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ('OPENROUTER_API_KEY', 'AZURE_OPENAI_API_KEY', 'QWEN_API_KEY'):
        os.environ.pop(key, None)
    manifest = dict(status='running', start=time.time(), pid=os.getpid(), algorithm=args.algorithm,
        runner_sha256=sha(__file__), dataset_sha256=sha(ROOT/'longmemeval.json'),
        selection='First 48 chronological source messages, no answer-conditioned selection; first 3 questions are retrieval probes only.',
        quality_scope='No cross-algorithm accuracy, complete-history or lifecycle claim.',
        deployment='Local Qwen3-8B construction/planning, Qwen3-32B answering, Qwen3-Embedding-0.6B; greedy, prefix cache enabled.',
        versions={p: importlib.metadata.version(p) for p in ['openai', 'torch', 'transformers', 'qdrant-client', 'lancedb']})
    write(out / 'manifest.json', manifest)
    try:
        history, data = source_history()
        write(out / 'history_private.json', history)
        manifest['history_sha256'] = sha(out/'history_private.json')
        install_call_audit(out)
        globals()[args.algorithm](out, history, data, manifest)
        calls = [json.loads(line) for line in (out/'llm_private.jsonl').read_text().splitlines()]
        assert calls and all(c['validated'] for c in calls), 'Upstream may swallow errors; audit rejects them'
        manifest.update(status='complete', end=time.time(), llm_calls=len(calls),
                        state_disk_bytes=sum(p.stat().st_size for p in (out/'state').rglob('*') if p.is_file()))
    except Exception:
        manifest.update(status='failed', end=time.time(), error=traceback.format_exc())
        raise
    finally:
        write(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
