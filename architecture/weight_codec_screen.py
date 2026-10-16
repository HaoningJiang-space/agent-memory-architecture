"""Bounded CPU screening of existing BF16 checkpoints; no model generation.

The exponent-window code is a TCA-TBE-inspired reference, not a new codec or
a ZipServ reproduction. Count metadata, 64-byte padding and raw escape blocks.
No sampled weight values or bitstreams leave the server.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

BLOCK=256
ALIGN=64


def window(words):
    histogram=np.bincount(((words>>7)&255),minlength=256)
    # 1..7 codes represent consecutive exponents; zero is an exact escape.
    return int(np.argmax(np.convolve(histogram,np.ones(7,dtype=int),mode='valid')))


def encode(words,lower):
    words=np.asarray(words,dtype='<u2');n=len(words)
    if n%8:raise ValueError('Bit-plane blocks must contain a multiple of eight weights')
    exponent=(words>>7)&255
    common=(exponent>=lower)&(exponent<lower+7)
    code=np.where(common,exponent-lower+1,0).astype(np.uint8)
    planes=b''.join(np.packbits((code>>i)&1,bitorder='little').tobytes() for i in range(3))
    low=(((words[common]>>8)&128)|(words[common]&127)).astype(np.uint8).tobytes()
    escapes=words[~common].tobytes()
    coded=struct.pack('<BBHI',lower,1,int((~common).sum()),n)+planes+low+escapes
    raw=struct.pack('<BBHI',0,0,0,n)+words.tobytes()
    blob=min((coded,raw),key=len)
    return blob+b'\0'*((-len(blob))%ALIGN)


def decode(blob):
    lower,mode,escapes,n=struct.unpack('<BBHI',blob[:8])
    if mode==0:return np.frombuffer(blob,dtype='<u2',count=n,offset=8).copy()
    code=np.zeros(n,dtype=np.uint16)
    for i in range(3):
        plane=np.frombuffer(blob,dtype=np.uint8,count=n//8,offset=8+i*n//8)
        code|=np.unpackbits(plane,bitorder='little').astype(np.uint16)<<i
    common=code!=0;assert int((~common).sum())==escapes
    offset=8+3*n//8
    low=np.frombuffer(blob,dtype=np.uint8,count=n-escapes,offset=offset).astype(np.uint16)
    values=np.empty(n,dtype='<u2')
    values[common]=((low&128)<<8)|((lower+code[common]-1)<<7)|(low&127)
    values[~common]=np.frombuffer(blob,dtype='<u2',count=escapes,offset=offset+n-escapes)
    return values


def address_table_bytes(blocks):
    """Two uint64 header words and one uint64 offset per independently read block."""
    if blocks < 1:
        raise ValueError('A matrix needs at least one block')
    size = 16 + 8 * blocks
    return size + (-size) % ALIGN


def headers(directory):
    found={};digests={}
    for p in sorted(directory.glob('*.safetensors')):
        with p.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0];raw=f.read(size)
        digests[p.name]=hashlib.sha256(raw).hexdigest()
        for name,h in json.loads(raw).items():
            if name!='__metadata__':found[name]=(p,8+size,h)
    return found,digests


def sample(directory):
    cfg=json.loads((directory/'config.json').read_text());tensors,hashes=headers(directory)
    layers=sorted(set((0,cfg['num_hidden_layers']//2,cfg['num_hidden_layers']-1)))
    names=[f'model.layers.{layer}.{op}.weight' for layer in layers for op in
           ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
            'mlp.gate_proj','mlp.up_proj','mlp.down_proj')]+['lm_head.weight']
    rows=[]
    for name in names:
        p,offset,h=tensors[name]
        if h['dtype']!='BF16':raise ValueError((name,h['dtype']))
        count=math.prod(h['shape']);length=4096
        positions=np.linspace(0,count-length,12,dtype=np.int64)
        data=np.memmap(p,dtype='<u2',mode='r',offset=offset+h['data_offsets'][0],shape=(count,))
        blocks=[np.array(data[i:i+length]) for i in positions];del data
        training=np.concatenate(blocks[::2]);heldout=np.concatenate(blocks[1::2]);lower=window(training)
        exponent=(heldout>>7)&255;hist=np.bincount(exponent,minlength=256);prob=hist[hist>0]/len(heldout)
        packed=[];raw_blocks=0
        for start in range(0,len(heldout),BLOCK):
            words=heldout[start:start+BLOCK];blob=encode(words,lower)
            assert np.array_equal(decode(blob),words)
            packed.append(len(blob));raw_blocks+=blob[1]==0
        coverage=float(np.mean((exponent>=lower)&(exponent<lower+7)))
        table_bytes=address_table_bytes(len(packed))
        rows.append(dict(tensor=name,shape=h['shape'],tensor_bytes=count*2,
            training_weights=len(training),heldout_weights=len(heldout),window_lower=lower,
            sample_sha256=hashlib.sha256(heldout.tobytes()).hexdigest(),
            exponent_entropy_bits=float(-sum(prob*np.log2(prob))),window_coverage=coverage,
            payload_bits_per_weight=3+8*coverage+16*(1-coverage),
            physical_encoded_bytes=sum(packed),baseline_raw_bytes=heldout.nbytes,
            encoded_to_raw_ratio=sum(packed)/heldout.nbytes,
            address_table_bytes=table_bytes,
            indexed_storage_bytes=sum(packed)+table_bytes,
            indexed_to_raw_ratio=(sum(packed)+table_bytes)/heldout.nbytes,
            blocks=len(packed),raw_blocks=raw_blocks,
            min_block_bytes=min(packed),max_block_bytes=max(packed),bit_exact=True))
    return dict(rows=rows,header_sha256=hashes,
        sampled_projection_weight_bytes=sum(r['tensor_bytes'] for r in rows),
        weighted_ratio=sum(r['tensor_bytes']*r['encoded_to_raw_ratio'] for r in rows)/sum(r['tensor_bytes'] for r in rows),
        weighted_indexed_ratio=sum(r['tensor_bytes']*r['indexed_to_raw_ratio'] for r in rows)/sum(r['tensor_bytes'] for r in rows),
        scope='Three layer positions plus LM head, twelve spaced windows per matrix; even windows train the exponent window, odd windows test. Projection subset only; no whole-model size claim.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads(args.models.read_text());rows=[]
    for name in ('Qwen/Qwen3-8B','Qwen/Qwen3-32B'):
        entry=manifest[name];rows.append(dict(model=name,revision=entry['revision'],**sample(Path(entry['path']))))
    out=dict(schema_version=2,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),block_weights=BLOCK,
        physical_alignment_bytes=ALIGN,metadata_bytes_per_block=8,
        address_table='16-byte header plus 8-byte offset per block, padded to 64 bytes per sampled matrix; charged separately from block payloads',
        numpy_version=np.__version__,models=rows,
        interpretation='Measured byte counts and exact reconstruction only. No decoder throughput, PPA, model-quality or inference-speed claim. Not a new compression algorithm.')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n')
    print([(r['model'],r['weighted_ratio']) for r in rows])


if __name__=='__main__':main()
