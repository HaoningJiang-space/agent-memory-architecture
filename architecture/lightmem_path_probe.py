"""Bounded LightMem path ablation with frozen LLMs and explicit flush repair.

Raw text stays remote. This restores encoder/topic processing, not the paper's
backbone, full benchmark, or quality result. Historical traces are untouched.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

from architecture import family_workload as family

ROOT = family.ROOT


def patch_flush_source(source):
    replacements = {
        'all_segments = self.senmem_buffer_manager.cut_with_segmenter(self.segmenter, self.text_embedder, force_segment)':
        'all_segments += self.senmem_buffer_manager.cut_with_segmenter(self.segmenter, self.text_embedder, force_segment)',
        'if not all_segments:\n            self.logger.debug':
        'if not all_segments and not (force_extract and self.shortmem_buffer_manager.buffer):\n            self.logger.debug',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('Pinned flush patch no longer matches upstream')
        source = source.replace(old, new)
    return source


def key(message):
    return hashlib.sha256(json.dumps(message, sort_keys=True, default=str).encode()).hexdigest()


def run(args, out, manifest):
    import torch
    native_path = ROOT / f'external/LightMem-{family.LIGHT_REV}'
    sys.path.insert(0, str(native_path / 'src'))
    import lightmem.memory.lightmem as native
    source_path = native_path / 'src/lightmem/memory/lightmem.py'
    original = source_path.read_text()
    patched = patch_flush_source(original)
    exec(compile(patched, str(source_path) + ':flush_adapter', 'exec'), native.__dict__)
    manifest['flush_adapter'] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        effective_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        changes=['Append forced sensory tail instead of replacing emitted segments.',
                 'Permit forced extraction of pending STM when no new segment is emitted.'])
    create = native.LightMemory.from_config
    compressor_source = json.loads((ROOT / 'outputs/compressor_source.json').read_text())
    manifest['encoder_source'] = compressor_source
    input_messages, extracted_messages = Counter(), Counter()
    linear = Counter()
    stage = ['initialization']

    def configured(cls, cfg):
        full = args.variant != 'flat'
        cfg = deepcopy(cfg)
        cfg.update(pre_compress=full, topic_segment=full, precomp_topic_shared=full)
        if full:
            cfg['pre_compressor'] = dict(model_name='llmlingua-2', configs=dict(
                llmlingua_config=dict(model_name=compressor_source['path'], device_map='cuda:0', use_llmlingua2=True),
                llmlingua2_config=dict(max_batch_size=50, max_force_token=100),
                compress_config=dict(instruction='', rate=.6, target_token=-1)))
            cfg['topic_segmenter'] = dict(model_name='llmlingua-2', configs=dict(layers=[8, 9, 10, 11]))
        system = create(cfg)
        manifest['initialization_end'] = time.time()
        # Native release hardcodes STM=512. Keep it for the paired path
        # ablation; the explicit 768 arm measures the paper-style buffer.
        system.shortmem_buffer_manager.max_tokens = args.stm_tokens
        manifest['effective_config'] = cfg
        manifest['stm_tokens'] = args.stm_tokens
        manifest['stm_tokenizer'] = type(system.shortmem_buffer_manager.tokenizer).__name__
        manifest['sensory_tokens'] = system.senmem_buffer_manager.max_tokens if full else None
        old_segments = system.shortmem_buffer_manager.add_segments

        def add_segments(segments, messages_use, force_extract=False):
            if not full:
                input_messages.update(key(m) for s in segments for m in s)
            result = old_segments(segments, messages_use, force_extract)
            extracted_messages.update(key(m) for batch in result[1] for s in batch for m in s)
            return result
        system.shortmem_buffer_manager.add_segments = add_segments
        if full:
            old_add = system.senmem_buffer_manager.add_messages
            def add_messages(messages, *rest, **kw):
                input_messages.update(key(m) for m in messages)
                return old_add(messages, *rest, **kw)
            system.senmem_buffer_manager.add_messages = add_messages
            encoder = system.compressor.inner_compressor.model
            # Attention tensors are required by the actual topic algorithm.
            if hasattr(encoder, 'set_attn_implementation'):
                encoder.set_attn_implementation('eager')
            manifest['encoder'] = dict(parameters=sum(p.numel() for p in encoder.parameters()),
                parameter_bytes=sum(p.numel()*p.element_size() for p in encoder.parameters()),
                dtype=str(next(encoder.parameters()).dtype), config=encoder.config.to_dict())
            original_forward = encoder.forward
            def forward(*a, **kw):
                ids = kw.get('input_ids', a[0] if a else None)
                tick = time.time()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                result = original_forward(*a, **kw)
                end.record(); end.synchronize()
                family.append(out/'encoder_forwards.jsonl', dict(stage=stage[0],
                    start=tick, end=time.time(), shape=list(ids.shape),
                    cuda_seconds=begin.elapsed_time(end)/1000,
                    attentions=bool(kw.get('output_attentions', False))))
                return result
            encoder.forward = forward
            for name, module in encoder.named_modules():
                if isinstance(module, torch.nn.Linear):
                    def hook(layer, a, name=name):
                        value = a[0]
                        m = value.numel()//value.shape[-1]
                        linear[(stage[0], name, m, layer.in_features, layer.out_features,
                                layer.weight.element_size())] += 1
                    module.register_forward_pre_hook(hook)
            for obj, attr, label in [(system.compressor, 'compress', 'compression'),
                                     (system.segmenter, 'sentence_level_attention', 'segmentation')]:
                original_method = getattr(obj, attr)
                def tracked(*a, _fn=original_method, _label=label, **kw):
                    previous = stage[0]; stage[0] = _label
                    try:
                        return _fn(*a, **kw)
                    finally:
                        stage[0] = previous
                setattr(obj, attr, tracked)
            # The native compressor swallows exceptions; retain an error
            # receipt and fail the experiment even if it falls back to raw text.
            compress_prompt = system.compressor.inner_compressor.compress_prompt
            def checked_compress(*a, **kw):
                try:
                    return compress_prompt(*a, **kw)
                except Exception as exc:
                    family.append(out/'encoder_errors.jsonl', dict(error=type(exc).__name__))
                    raise
            system.compressor.inner_compressor.compress_prompt = checked_compress
        original_add_memory = system.add_memory
        def add_memory(messages, **kw):
            if full and kw.get('force_extract'):
                kw['force_segment'] = True
            result = original_add_memory(messages, **kw)
            if kw.get('force_extract'):
                assert not system.shortmem_buffer_manager.buffer, 'Unflushed STM'
                if full:
                    assert not system.senmem_buffer_manager.buffer, 'Unflushed sensory buffer'
                    assert not system.senmem_buffer_manager.big_buffer, 'Unconsumed sensory input'
                assert input_messages == extracted_messages, 'Message loss or duplication at extraction boundary'
                manifest['message_conservation'] = dict(input_messages=sum(input_messages.values()),
                    extracted_messages=sum(extracted_messages.values()), equal=True)
                if (out/'encoder_errors.jsonl').exists():
                    raise ValueError('Compressor fallback occurred')
            return result
        system.add_memory = add_memory
        return system

    native.LightMemory.from_config = classmethod(configured)
    history, data = family.source_history()
    family.write(out/'history_private.json', history)
    manifest['history_sha256'] = family.sha(out/'history_private.json')
    family.install_call_audit(out)
    family.lightmem(out, history, data, manifest)
    manifest['encoder_linears'] = [dict(stage=s, name=n, M=m, K=k, N=v, bytes_per_weight=b, calls=c,
        flops=2*m*k*v*c, one_pass_weight_bytes=b*k*v*c) for (s,n,m,k,v,b),c in linear.items()]
    assert manifest['message_conservation']['input_messages'] == 48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['flat', 'full'], required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--stm-tokens', type=int, choices=[512, 768], default=512)
    args = parser.parse_args()
    assert args.session.replace('_','').isalnum()
    family.SESSION = args.session
    out = ROOT/'outputs'/args.session/'lightmem'
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HOME=str(ROOT/'hf'), HF_HUB_OFFLINE='1',
        TIKTOKEN_CACHE_DIR=str(ROOT/'tiktoken-cache'), NLTK_DATA=str(ROOT/'nltk_data'),
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    manifest = dict(status='running', start=time.time(), pid=os.getpid(), algorithm='lightmem',
        variant=args.variant, runner_sha256=family.sha(__file__),
        dataset_sha256=family.sha(ROOT/'longmemeval.json'),
        selection='Same first 48 chronological messages and first 3 retrieval probes as family runs.',
        quality_scope='Path ablation only; probes not guaranteed answerable; no quality-equivalence claim.',
        deployment='Qwen3-8B / Qwen3-32B BF16; embedding Qwen3-Embedding-0.6B; native encoder paths restored.',
        versions={p: importlib.metadata.version(p) for p in ['torch','transformers','llmlingua','openai','nltk']})
    family.write(out/'manifest.json', manifest)
    try:
        run(args, out, manifest)
        calls = [json.loads(s) for s in (out/'llm_private.jsonl').read_text().splitlines()]
        assert calls and all(c['validated'] for c in calls)
        operations = [json.loads(s) for s in (out/'operations.jsonl').read_text().splitlines()]
        assert all(r['status']=='ok' for r in operations)
        manifest.update(status='complete', end=time.time(), llm_calls=len(calls),
            state_disk_bytes=sum(p.stat().st_size for p in (out/'state').rglob('*') if p.is_file()))
    except Exception:
        manifest.update(status='failed', end=time.time(), error=traceback.format_exc())
        raise
    finally:
        family.write(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
