"""Exploratory BF16 tensor/dataflow model, not a calibrated NPU simulator.

Requests are replayed independently. Array waves, finite double-buffered SRAM,
OS/WS traffic, causal GQA attention, DRAM service and capacity are explicit.
No API latency is scaled into a hardware prediction.
"""
from dataclasses import dataclass, asdict, replace
from functools import lru_cache
import math

from workload_accounting import decoder_shape

GIB = 2**30
MIB = 2**20


def ceildiv(a, b):
    return (a + b - 1) // b


@dataclass(frozen=True)
class Hardware:
    name: str
    arrays: int
    rows: int
    cols: int
    sram_mib: int
    memory: str
    packages: int
    capacity_bytes: int
    bandwidth: float
    latency: float
    read_pj_bit: float
    write_pj_bit: float
    memory_background_w: float
    area_mm2: float
    cost_units: float
    shoreline_mm: float
    peak_power_w: float
    frequency: float = 1e9
    bandwidth_efficiency: float = .8
    mac_pj: float = .682
    sram_pj_bit: float = .1
    base_power_w: float = 30
    sram_bandwidth_per_array: float = 512e9
    cycle_mode: str = 'scheduled'
    fill_mode: str = 'per_k_tile'
    freeze_mapping_bandwidth: float = 0

    @property
    def macs(self):
        return self.arrays*self.rows*self.cols

    @property
    def static_w(self):
        # Conventional SRAM background power is an explicit midpoint scenario.
        return self.base_power_w + self.memory_background_w + self.sram_mib*MIB/1e9*30


def candidates(spec, cost_multiplier=1., area_multiplier=1., mac_multiplier=1.):
    """Coupled per-package bandwidth/capacity/shoreline, shared budget envelope."""
    result, rejected = [], []
    p, budget = spec['logic_assumptions'], spec['budget']
    for memory, packages in spec['memory_options']:
        mem = spec['memory_technology'][memory]
        for arrays in spec['array_counts']:
            for rows, cols in spec['array_shapes']:
                for sram in spec['sram_mib']:
                    macs = arrays*rows*cols
                    shore = packages*mem['shoreline_mm']
                    area = (p['fixed_area_mm2'] + area_multiplier*(macs*p['mm2_per_mac'] +
                            sram*p['mm2_per_sram_mib']) + shore*p['phy_depth_mm'])
                    cost = area*p['cost_per_mm2'] + packages*mem['cost_units']*cost_multiplier
                    bg = packages*mem['capacity_GB']*mem['background_w_GB']
                    static = p['base_power_w']+bg+sram*MIB/1e9*30
                    # Conservative full-read port + all MACs active; SRAM dynamic
                    # power is a separate envelope. This is not a measured TDP.
                    peak = (static+macs*1e9*p['mac_pj']*mac_multiplier*1e-12+
                            packages*mem['bandwidth_GBs']*1e9*8*mem['write_pj_bit']*1e-12+
                            p['sram_dynamic_reserve_w'])
                    name = f'{memory}{packages}_a{arrays}_{rows}x{cols}_s{sram}'
                    violations = [key for key, value in [('cost_units',cost),('area_mm2',area),
                                  ('shoreline_mm',shore),('power_w',peak)] if value>budget[key]]
                    if violations:
                        rejected.append(dict(name=name,violations=violations))
                        continue
                    result.append(Hardware(name,arrays,rows,cols,sram,memory,packages,
                        int(packages*mem['capacity_GB']*1e9),packages*mem['bandwidth_GBs']*1e9,
                        mem['latency_ns']*1e-9,mem['read_pj_bit'],mem['write_pj_bit'],bg,
                        area,cost,shore,peak,mac_pj=p['mac_pj']*mac_multiplier,
                        base_power_w=p['base_power_w']))
    return result, rejected


def service(hw, cycles, reads, writes, flops, sram_bytes, transactions=1, cycle_terms=None):
    if hw.cycle_mode == 'ideal':
        cycles = flops/(2*hw.macs)
        cycle_terms = {'useful':cycles}
    elif hw.cycle_mode != 'scheduled':
        raise ValueError(hw.cycle_mode)
    compute = cycles/hw.frequency
    # 32 independent outstanding bulk transfers is a declared controller
    # scenario, not a measured MLP or a row-buffer timing simulation.
    memory = (reads+writes)/(hw.bandwidth*hw.bandwidth_efficiency)+ceildiv(transactions,32)*hw.latency
    sram_time = sram_bytes/(hw.arrays*hw.sram_bandwidth_per_array)
    seconds = max(compute, memory, sram_time)  # optimistic inter-tile overlap
    energy = (flops/2*hw.mac_pj*1e-12 + 8*(reads*hw.read_pj_bit+writes*hw.write_pj_bit)*1e-12 +
              sram_bytes*8*hw.sram_pj_bit*1e-12 + hw.static_w*seconds)
    terms = {f'array_{key}_seconds':(cycle_terms or {}).get(key,0)/hw.frequency
             for key in CYCLE_KEYS}
    limiting = max(('compute','memory','sram'),key=lambda key:dict(compute=compute,memory=memory,sram=sram_time)[key])
    terms.update({f'limited_by_{key}_seconds':seconds if key==limiting else 0
                  for key in ('compute','memory','sram')})
    return dict(seconds=seconds,energy_j=energy,flops=flops,read_bytes=reads,write_bytes=writes,
                sram_bytes=sram_bytes,compute_seconds=compute,memory_seconds=memory,
                sram_seconds=sram_time,**terms)


CYCLE_KEYS = ('useful','m_padding','n_padding','tile_padding','wave_tail','fill_drain','attention_mask_tail')


def add(*records):
    keys = ('seconds','energy_j','flops','read_bytes','write_bytes','sram_bytes',
            'compute_seconds','memory_seconds','sram_seconds')+tuple(f'array_{k}_seconds' for k in CYCLE_KEYS)+tuple(
                f'limited_by_{k}_seconds' for k in ('compute','memory','sram'))
    return {k:sum(r.get(k,0) for r in records) for k in keys}


def scale(record, count):
    return {k:v*count for k,v in add(record).items()}


@lru_cache(maxsize=200000)
def gemm(hw, m, k, n):
    """C[M,N] = A[M,K] B[K,N]; tiled OS or WS selected per operator.

OS keeps FP32 C across K tiles. WS reads each weight once and spills FP32
partial C between K tiles. Both charge A reloads across N tiles. No cross-op
activation reuse or persistent weight cache is silently assumed.
"""
    options = []
    local = hw.sram_mib*MIB/hw.arrays
    for tm in sorted(set([min(m,x) for x in (hw.rows,128,512,2048)])):
        for tn in sorted(set([min(n,x) for x in (hw.cols,512,2048)])):
            for tk in (128,512):
                tk = min(k,tk)
                footprint = 2*(2*tm*tk+2*tk*tn)+4*tm*tn
                if footprint>local:
                    continue
                # A tile cannot use a physical array dimension larger than
                # its data; count padding and incomplete array waves explicitly.
                micro_m = ceildiv(m,tm)*ceildiv(tm,hw.rows)
                micro_n = ceildiv(n,tn)*ceildiv(tn,hw.cols)
                waves = ceildiv(micro_m*micro_n,hw.arrays)
                fills = ceildiv(k,tk) if hw.fill_mode=='per_k_tile' else 1
                cycles = waves*(k+fills*(hw.rows+hw.cols-2))
                padded_m,padded_n=ceildiv(m,hw.rows)*hw.rows,ceildiv(n,hw.cols)*hw.cols
                cycle_terms=dict(useful=m*k*n/hw.macs,
                    m_padding=(padded_m-m)*k*n/hw.macs,
                    n_padding=padded_m*k*(padded_n-n)/hw.macs,
                    tile_padding=(micro_m*micro_n*hw.rows*hw.cols-padded_m*padded_n)*k/hw.macs,
                    wave_tail=(waves*hw.arrays-micro_m*micro_n)*k/hw.arrays,
                    fill_drain=waves*fills*(hw.rows+hw.cols-2))
                ar = 2*m*k*ceildiv(n,tn)
                bw = 2*k*n
                sram = 2*k*(m*ceildiv(n,hw.cols)+n*ceildiv(m,hw.rows))+4*m*n
                for flow, reads, writes in (
                    ('OS', ar+bw*ceildiv(m,tm),2*m*n),
                    ('WS', ar+bw+4*m*n*(ceildiv(k,tk)-1),
                           4*m*n*(ceildiv(k,tk)-1)+2*m*n)):
                    row = service(hw,cycles,reads,writes,2*m*k*n,sram,
                                  transactions=ceildiv(m,tm)*ceildiv(n,tn),cycle_terms=cycle_terms)
                    row.update(dataflow=flow,tile=[tm,tk,tn],buffer_bytes=footprint,
                               micro_tiles=micro_m*micro_n,array_waves=waves,
                               k_tiles=ceildiv(k,tk),fill_events=waves*fills)
                    options.append(row)
    if not options:
        raise ValueError(f'No legal tile for {hw.name}, {(m,k,n)}')
    if hw.freeze_mapping_bandwidth:
        original = replace(hw,bandwidth=hw.freeze_mapping_bandwidth,freeze_mapping_bandwidth=0,
                           cycle_mode='scheduled',fill_mode='per_k_tile')
        frozen = gemm(original,m,k,n)
        return next(r for r in options if r['tile']==frozen['tile'] and r['dataflow']==frozen['dataflow'])
    return min(options,key=lambda x:(x['seconds'],x['energy_j']))


@lru_cache(maxsize=100000)
def attention(hw, length, heads, kv_heads, dim, batch=1, decode=False):
    """Streaming causal attention. GQA KV reused across heads in one group.

Prefill reads each preceding K/V block once per query block and KV group.
Scores stay on chip; diagonal tiles execute masked entries too. Decode loads
the context once per group. Softmax/vector work is not yet modeled.
"""
    group = heads//kv_heads
    tile = 128
    while (2*(2*tile*dim*group+4*tile*dim)+4*tile*dim*group+4*tile*tile*group
           > hw.sram_mib*MIB/hw.arrays) and tile>16:
        tile//=2
    q = 1 if decode else length
    blocks = ceildiv(length,tile)
    pairs = length if decode else length*(length+1)//2
    if decode:
        tiles = blocks
        reads = 4*length*kv_heads*dim*batch+2*heads*dim*batch
    else:
        tiles = blocks*(blocks+1)//2
        # Includes padded diagonal and final K/V block reads: conservative.
        reads = 4*tiles*tile*kv_heads*dim*batch+2*length*heads*dim*batch
    writes = 2*q*heads*dim*batch
    qm = 1 if decode else tile
    # QK and PV, same head tasks, scheduled as waves across physical arrays.
    qk_waves = ceildiv(batch*heads*tiles*ceildiv(qm,hw.rows)*ceildiv(tile,hw.cols),hw.arrays)
    pv_waves = ceildiv(batch*heads*tiles*ceildiv(qm,hw.rows)*ceildiv(dim,hw.cols),hw.arrays)
    cycles = qk_waves*(dim+hw.rows+hw.cols-2)+pv_waves*(tile+hw.rows+hw.cols-2)
    flops = 4*batch*heads*dim*pairs
    tasks=batch*heads*tiles
    terms=dict(useful=flops/(2*hw.macs),m_padding=0,n_padding=0,wave_tail=0,fill_drain=0,
               attention_mask_tail=(2*tasks*qm*dim*tile-flops/2)/hw.macs)
    for k,n,waves in [(dim,tile,qk_waves),(tile,dim,pv_waves)]:
        pm,pn=ceildiv(qm,hw.rows)*hw.rows,ceildiv(n,hw.cols)*hw.cols
        micros=tasks*ceildiv(qm,hw.rows)*ceildiv(n,hw.cols)
        terms['m_padding']+=tasks*(pm-qm)*k*n/hw.macs
        terms['n_padding']+=tasks*pm*k*(pn-n)/hw.macs
        terms['wave_tail']+=(waves*hw.arrays-micros)*k/hw.arrays
        terms['fill_drain']+=waves*(hw.rows+hw.cols-2)
    sram = batch*heads*tiles*2*(qm*dim+2*tile*dim+qm*tile)
    row = service(hw,cycles,reads,writes,flops,sram,transactions=tiles*batch*kv_heads,cycle_terms=terms)
    row['tile'] = tile
    row.update(qk_array_waves=qk_waves,pv_array_waves=pv_waves,head_block_tasks=tasks)
    return row


def decoder(hw, config, prompt, completion, embedding=False, batch=1):
    h, d = config['hidden_size'],config['head_dim']
    q, kv = config['num_attention_heads']*d,config['num_key_value_heads']*d
    ff, layers = config['intermediate_size'],config['num_hidden_layers']
    shapes = [(h,q),(h,kv),(h,kv),(q,h),(h,ff),(h,ff),(ff,h)]
    pre_gemm=scale(add(*(gemm(hw,batch*prompt,k,n) for k,n in shapes)),layers)
    pre_attn=scale(attention(hw,prompt,config['num_attention_heads'],config['num_key_value_heads'],d,batch),layers)
    if not embedding:
        pre_gemm = add(pre_gemm,gemm(hw,batch,h,config['vocab_size']))
    prefill=add(pre_gemm,pre_attn)
    steps = 0 if embedding else completion-1
    dense = scale(add(*(gemm(hw,batch,k,n) for k,n in shapes)),layers)
    if not embedding:
        dense = add(dense,gemm(hw,batch,h,config['vocab_size']))
    # Context lengths differ over decode. Sum exact logical/traffic terms;
    # array waves use the central length, bounded by the first/last lengths.
    if steps:
        middle = prompt+(steps+1)//2
        att = attention(hw,middle,config['num_attention_heads'],config['num_key_value_heads'],d,batch,True)
        dec_attn=scale(att,layers*steps)
    else:
        dec_attn=add()
    dec_gemm=scale(dense,steps)
    decoding=add(dec_gemm,dec_attn)
    return dict(prefill=prefill,decode=decoding,total=add(prefill,decoding),
                components=dict(prefill_gemm=pre_gemm,prefill_attention=pre_attn,
                                decode_gemm=dec_gemm,decode_attention=dec_attn),
                kv_bytes=0 if embedding else decoder_shape(config)['kv_bytes_per_token']*batch*(prompt+steps),
                workspace_bytes=4*GIB+2*batch*prompt*(h+2*ff),
                decode_attention_approximation='central context; exact endpoints differ by at most output length')


def hardware_dict(hw):
    return asdict(hw)
