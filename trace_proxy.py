"""Local-only OpenAI router and raw request trace; no external inference APIs."""
import json
from pathlib import Path
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

ROOT = Path(__file__).resolve().parent
app = FastAPI()
PORTS = {'Qwen/Qwen3-8B': 8031, 'Qwen/Qwen3-32B': 8032,
         'Qwen/Qwen3-Embedding-0.6B': 8033}

def save(row):
    with (ROOT / 'outputs' / 'api_trace.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')

@app.get('/health')
def health():
    return {'router': 'ready'}

@app.post('/v1/{path:path}')
async def route(path: str, request: Request):
    body = await request.json()
    model = body.get('model')
    if model not in PORTS:
        return Response('Unknown local model', status_code=400)
    if path == 'chat/completions':
        body['chat_template_kwargs'] = {'enable_thinking': False}
        # vLLM 0.10 rejects redundant schema + generic JSON-object guidance.
        # The explicit schema already requires a JSON object.
        if body.get('guided_json') and body.get('response_format', {}).get('type') == 'json_object':
            body.pop('response_format')
    marker = ROOT / 'run_context.json'
    context = json.loads(marker.read_text()) if marker.exists() else {}
    row = {'id': str(uuid.uuid4()), 'model': model, 'path': path,
           'context': context, 'start': time.time(), 'request': body}
    client = httpx.AsyncClient(timeout=httpx.Timeout(1800))
    try:
        upstream = await client.send(client.build_request(
            'POST', f'http://127.0.0.1:{PORTS[model]}/v1/{path}', json=body), stream=True)
    except Exception as exc:
        row.update(end=time.time(), error=str(exc))
        save(row)
        await client.aclose()
        raise
    row['status'] = upstream.status_code
    if body.get('stream') and upstream.status_code == 200:
        async def relay():
            buffer = b''
            content = []
            try:
                async for chunk in upstream.aiter_bytes():
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':
                            continue
                        payload = json.loads(line[6:])
                        if payload.get('usage'):
                            row['usage'] = payload['usage']
                        for choice in payload.get('choices', []):
                            delta = choice.get('delta', {})
                            if delta.get('content') or delta.get('tool_calls'):
                                row.setdefault('first_token', time.time())
                            if delta.get('content'):
                                content.append(delta['content'])
                            if choice.get('finish_reason'):
                                row['finish_reason'] = choice['finish_reason']
                    yield chunk
            finally:
                row.update(end=time.time(), response_text=''.join(content))
                save(row)
                await upstream.aclose()
                await client.aclose()
        return StreamingResponse(relay(), media_type='text/event-stream')
    raw = await upstream.aread()
    row['end'] = time.time()
    try:
        payload = json.loads(raw)
        row['usage'] = payload.get('usage')
        if path == 'embeddings':
            row['embedding_count'] = len(payload.get('data', []))
        else:
            row['response'] = payload
    except ValueError:
        row['response_raw'] = raw.decode(errors='replace')[:4096]
    save(row)
    await upstream.aclose()
    await client.aclose()
    return Response(raw, status_code=upstream.status_code, media_type='application/json')
