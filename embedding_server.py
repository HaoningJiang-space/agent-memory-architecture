"""Pinned HF last-token embedding endpoint with explicit kernel batches."""
import base64
import json
from pathlib import Path
import threading
import time

from fastapi import FastAPI
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3-Embedding-0.6B'
info = json.loads((ROOT / 'models.json').read_text())[MODEL]
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(info['path'], padding_side='left', local_files_only=True)
model = AutoModel.from_pretrained(info['path'], torch_dtype=torch.bfloat16,
                                attn_implementation='sdpa', local_files_only=True).cuda().eval()
app = FastAPI()
lock = threading.Lock()

@app.get('/health')
def health():
    return {'model': MODEL}

@app.post('/v1/embeddings')
def embed(body: dict):
    texts = body['input']
    if isinstance(texts, str):
        texts = [texts]
    assert all(isinstance(t, str) for t in texts), 'Text input required'
    vectors, token_count = [], 0
    with lock, torch.inference_mode():
        for offset in range(0, len(texts), 32):
            sub = texts[offset:offset+32]
            # Refuse silent truncation: the native workload must supply valid lengths.
            cpu = tokenizer(sub, padding=True, truncation=False, return_tensors='pt')
            lengths = cpu['attention_mask'].sum(dim=1).tolist()
            assert max(lengths) <= 32768
            tensors = {k: v.cuda() for k,v in cpu.items()}
            start = time.time()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            hidden = model(**tensors, use_cache=False).last_hidden_state
            values = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
            b.record()
            b.synchronize()
            vectors.extend(values.cpu().numpy())
            row = {'start': start, 'end': time.time(), 'batch': len(sub), 'lengths': lengths,
                   'padded_length': tensors['input_ids'].shape[1],
                   'cuda_seconds': a.elapsed_time(b)/1000, 'api_count': len(texts), 'offset': offset}
            with (ROOT/'outputs'/'embedding_kernels.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            token_count += sum(lengths)
    data = []
    for i, vector in enumerate(vectors):
        v = base64.b64encode(np.asarray(vector, dtype='<f4').tobytes()).decode() if body.get('encoding_format') == 'base64' else vector.tolist()
        data.append({'object':'embedding', 'index':i, 'embedding':v})
    return {'object':'list', 'model':MODEL, 'data':data,
            'usage': {'prompt_tokens':token_count, 'total_tokens':token_count}}
