"""Exact stored BF16 weights and config-derived KV capacity, with explicit units."""
import json
from pathlib import Path
import pynvml

root=Path(__file__).resolve().parent
models=json.loads((root/'models.json').read_text())
rows=[]
for name,info in models.items():
    p=Path(info['path'])
    cfg=json.loads((p/'config.json').read_text())
    index=p/'model.safetensors.index.json'
    if index.exists():
        byte_count=json.loads(index.read_text())['metadata']['total_size']
    else:
        from safetensors import safe_open
        byte_count=0
        for file in p.glob('*.safetensors'):
            with safe_open(file,framework='pt',device='cpu') as f:
                for key in f.keys():
                    tensor=f.get_slice(key)
                    import math
                    dtype=tensor.get_dtype()
                    assert dtype=='BF16',dtype
                    byte_count+=math.prod(tensor.get_shape())*2
    kv=2*cfg['num_hidden_layers']*cfg['num_key_value_heads']*cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])*2
    rows.append(dict(model=name,revision=info['revision'],weight_bytes=byte_count,
                     weight_GiB=byte_count/2**30,KV_bytes_per_token_BF16=kv,
                     KV_GiB_at_32768=kv*32768/2**30,
                     note='embedding does not retain generation KV' if 'Embedding' in name else 'KV formula excludes allocator/workspace'))
pynvml.nvmlInit()
gpus=[dict(index=i,total_bytes=pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total) for i in range(pynvml.nvmlDeviceGetCount())]
out=dict(models=rows,gpus=gpus,all_model_weight_bytes=sum(r['weight_bytes'] for r in rows),
         scope='weights and config-derived KV only; no measured DRAM traffic or fitting claim')
(root/'outputs/model_capacity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
