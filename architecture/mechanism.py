"""Counted K-shard mapping on a unified array; costs are explicit, unsynthesized.

The 7cd9029 model stays frozen. This extension conserves its baseline service
and categorizes DRAM traffic, then adds a schedulable M=1 mapping. Each array
splits rows into K groups, writes FP32 local partials and serially reduces them.
Bank and reduction service are bounded; no new off-chip bandwidth is free.
"""
from dataclasses import dataclass,replace,asdict
from functools import lru_cache
import math

from architecture import model as base

TRAFFIC=('weight_read_bytes','kv_read_bytes','activation_read_bytes','partial_read_bytes',
         'activation_write_bytes','partial_write_bytes')
EXTRA=TRAFFIC+('reduction_adds','reduction_link_bytes','input_distribution_bytes',
              'reduction_seconds','limited_by_reduction_seconds','k_shard_operations')


def add(*rows):
    return {**base.add(*rows),**{k:sum(r.get(k,0) for r in rows) for k in EXTRA}}


def scale(row,count):
    return {k:v*count for k,v in add(row).items()}


def linear(hw,m,k,n):
    r=dict(base.gemm(hw,m,k,n));tm,tk,tn=r['tile']
    weight=2*k*n*(base.ceildiv(m,tm) if r['dataflow']=='OS' else 1)
    partial=0 if r['dataflow']=='OS' else 4*m*n*(base.ceildiv(k,tk)-1)
    r.update(weight_read_bytes=weight,kv_read_bytes=0,activation_read_bytes=r['read_bytes']-weight-partial,
             partial_read_bytes=partial,activation_write_bytes=2*m*n,partial_write_bytes=partial)
    return r


def attention(hw,length,heads,kv_heads,dim,batch=1,decode=False):
    r=dict(base.attention(hw,length,heads,kv_heads,dim,batch,decode))
    activation=2*(1 if decode else length)*heads*dim*batch
    r.update(weight_read_bytes=0,kv_read_bytes=r['read_bytes']-activation,activation_read_bytes=activation,
             partial_read_bytes=0,activation_write_bytes=r['write_bytes'],partial_write_bytes=0)
    return r


@dataclass(frozen=True)
class KShard:
    groups: int=16
    reduction_lanes: int=16
    adder_latency_cycles: int=3
    sync_cycles: int=8
    reduction_link_bytes_cycle: int=64
    bank_skew: float=1
    transfer_granularity: str='k_shard_stream'
    # Sensitivities, not calibrated PPA coefficients.
    overhead_multiplier: float=1
    clock_drop_per_level: float=.005
    adder_area_mac_ratio: float=.5
    mux_mm2_bit: float=2e-7
    address_mm2: float=.002
    adder_pj: float=.34
    wire_pj_bit: float=.2
    mux_pj_mac: float=.02


def provision(hw,scheme,spec):
    """Charge interfaces, adders, control, buffers and a global clock penalty."""
    assert hw.rows%scheme.groups==0 and scheme.groups>=2
    g=scheme.groups;depth=math.ceil(math.log2(g));a=hw.arrays
    adders=a*scheme.reduction_lanes*(g-1)
    mux_bits=2*16*hw.macs
    addresses=a*g
    # Double-buffered FP32 partials are carved from existing SRAM, not free.
    buffer_bytes=a*2*4*g*hw.cols
    logic_area=(adders*spec['logic_assumptions']['mm2_per_mac']*scheme.adder_area_mac_ratio+
                mux_bits*scheme.mux_mm2_bit+addresses*scheme.address_mm2)*scheme.overhead_multiplier
    clock=1/(1+scheme.clock_drop_per_level*depth)
    full_dynamic=(adders*scheme.adder_pj*1e-12*hw.frequency+
                  hw.macs*scheme.mux_pj_mac*1e-12*hw.frequency)*scheme.overhead_multiplier
    leakage=logic_area*.1
    cost=hw.cost_units+logic_area*spec['logic_assumptions']['cost_per_mm2']
    # Conservatively retain base peak MAC power despite lower clock, and reserve
    # full link toggling at every array; this is an envelope, not measured TDP.
    link_peak=a*scheme.reduction_link_bytes_cycle*hw.frequency*8*scheme.wire_pj_bit*1e-12
    peak=hw.peak_power_w+full_dynamic+leakage+link_peak
    modified=replace(hw,name=hw.name+f'_k{g}',frequency=hw.frequency*clock,
        sram_bandwidth_per_array=hw.sram_bandwidth_per_array*clock,
        area_mm2=hw.area_mm2+logic_area,cost_units=cost,peak_power_w=peak,
        base_power_w=hw.base_power_w+leakage,fill_mode='continuous_k')
    charges=dict(adders=adders,mux_bits=mux_bits,address_generators=addresses,
        partial_buffer_bytes=buffer_bytes,added_logic_area_mm2=logic_area,clock_factor=clock,
        added_full_dynamic_w=full_dynamic,link_peak_w=link_peak,added_leakage_w=leakage,
        area_mm2=modified.area_mm2,cost_units=cost,power_envelope_w=peak,
        banks_per_array=32,bytes_per_bank_cycle=16,
        scope='Declared 1 GHz bank-port envelope and unsynthesized logic coefficients; no routed timing or measured area.')
    violations=[k for k,v in [('area_mm2',modified.area_mm2),('cost_units',cost),
                ('power_w',peak),('shoreline_mm',modified.shoreline_mm)] if v>spec['budget'][k]]
    return modified,charges,violations


@lru_cache(maxsize=100000)
def sharded(hw,scheme,m,k,n):
    """K groups share N output tiles; their partials meet at a local tree.

Only M=1 projections use this path. Attention retains the original schedule.
Weight layout is [N tile, K group, K offset, column], balanced over local
banks. At most the existing SRAM bandwidth can be consumed; bank skew stalls
both stages. Full SRAM/DRAM service can overlap MACs, but reduction is serial.
"""
    assert m==1 and hw.rows%scheme.groups==0
    g,c=scheme.groups,hw.cols;subrows=hw.rows//g
    kp=base.ceildiv(k,g);tiles=base.ceildiv(n,c);waves=base.ceildiv(tiles,hw.arrays)
    local=hw.sram_mib*base.MIB/hw.arrays
    reserved=2*4*g*c
    legal=[t for t in (128,512) if 2*(2*g*min(kp,t)+2*g*min(kp,t)*c)+reserved<=local]
    if not legal: return None
    tk=min(kp,max(legal))
    cycles=waves*(kp+subrows+c-2)
    cycle_terms=dict(useful=k*n/hw.macs,m_padding=(subrows-1)*k*n/hw.macs,
        n_padding=subrows*k*(tiles*c-n)/hw.macs,
        tile_padding=subrows*(g*kp-k)*tiles*c/hw.macs,
        wave_tail=(waves*hw.arrays-tiles)*kp/hw.arrays,
        fill_drain=waves*(subrows+c-2))
    weight=2*k*n;activation=2*k*tiles;writes=2*n
    # Same read-service convention as baseline; additional broadcast traffic
    # goes on local wires and is explicitly charged in energy below.
    local_bytes=weight+activation+4*g*n
    traffic=weight+activation
    # Continuous K consumes double buffers without forcing a new external
    # transaction at every refill. One contiguous stream per output tile/shard.
    # The buffer_tile control deliberately removes this DMA coalescing.
    transactions=tiles*g
    if scheme.transfer_granularity=='buffer_tile': transactions*=base.ceildiv(kp,tk)
    else: assert scheme.transfer_granularity=='k_shard_stream'
    # Bank serialization reduces effective service, not the bytes requested.
    service_hw=replace(hw,sram_bandwidth_per_array=hw.sram_bandwidth_per_array/scheme.bank_skew)
    r=base.service(service_hw,cycles,traffic,writes,2*k*n,local_bytes,transactions=transactions,cycle_terms=cycle_terms)
    depth=math.ceil(math.log2(g))
    tree_cycles=waves*(base.ceildiv(c,scheme.reduction_lanes)+depth*scheme.adder_latency_cycles+scheme.sync_cycles)
    reduction_bytes=4*n*(2*g-2)
    tree_time=tree_cycles/hw.frequency
    link_time=reduction_bytes/(hw.arrays*scheme.reduction_link_bytes_cycle*hw.frequency)
    reduction_bank_bytes=4*g*n+4*n
    bank_time=reduction_bank_bytes*scheme.bank_skew/(hw.arrays*hw.sram_bandwidth_per_array)
    red=max(tree_time,link_time,bank_time)
    adds=(g-1)*n
    broadcast=2*k*n
    r['seconds']+=red
    r['sram_bytes']+=reduction_bank_bytes
    r['sram_seconds']+=bank_time
    r['energy_j']+=red*hw.static_w+(adds*scheme.adder_pj+8*(reduction_bytes+broadcast)*scheme.wire_pj_bit+
                                  k*n*scheme.mux_pj_mac)*1e-12*scheme.overhead_multiplier
    r['energy_j']+=8*reduction_bank_bytes*hw.sram_pj_bit*1e-12
    # Tree's FP32 partial buffers remain local; no silent off-chip spill.
    r.update(weight_read_bytes=weight,kv_read_bytes=0,activation_read_bytes=activation,partial_read_bytes=0,
        activation_write_bytes=writes,partial_write_bytes=0,reduction_adds=adds,reduction_link_bytes=reduction_bytes,
        input_distribution_bytes=broadcast,reduction_seconds=red,limited_by_reduction_seconds=red,k_shard_operations=1,
        dataflow='K_SHARD',tile=[1,tk,c],buffer_bytes=2*(2*g*tk+2*g*tk*c)+reserved,
        k_groups=g,k_partition=kp,array_waves=waves,
        dram_transfer_groups=transactions,
        tree_seconds=tree_time,reduction_link_seconds=link_time,reduction_bank_seconds=bank_time,
        peak_weight_supply_bytes_s_per_array=2*g*c*hw.frequency,
        actual_port_budget_bytes_s_per_array=hw.sram_bandwidth_per_array)
    return r


def mapped_linear(hw,m,k,n,scheme=None):
    normal=linear(hw,m,k,n)
    if scheme is not None:
        normal['energy_j']+=normal['flops']/2*scheme.mux_pj_mac*1e-12*scheme.overhead_multiplier
    if scheme is None or m!=1: return normal
    other=sharded(hw,scheme,m,k,n)
    if other is None: return normal
    return min((normal,other),key=lambda r:(r['seconds'],r['energy_j']))


def evaluate(hw,config,prompt,completion,scheme=None):
    """Same first-token convention and causal attention as frozen decoder()."""
    h,d=config['hidden_size'],config['head_dim']
    q,kv=config['num_attention_heads']*d,config['num_key_value_heads']*d
    ff,layers=config['intermediate_size'],config['num_hidden_layers']
    shapes=[(h,q),(h,kv),(h,kv),(q,h),(h,ff),(h,ff),(ff,h)]
    pg=scale(add(*(mapped_linear(hw,prompt,k,n,scheme) for k,n in shapes)),layers)
    pg=add(pg,mapped_linear(hw,1,h,config['vocab_size'],scheme))
    pa=scale(attention(hw,prompt,config['num_attention_heads'],config['num_key_value_heads'],d),layers)
    steps=completion-1
    dg=scale(add(*(mapped_linear(hw,1,k,n,scheme) for k,n in shapes)),layers)
    dg=scale(add(dg,mapped_linear(hw,1,h,config['vocab_size'],scheme)),steps)
    da=scale(attention(hw,prompt+(steps+1)//2,config['num_attention_heads'],config['num_key_value_heads'],d,decode=True),layers*steps) if steps else add()
    components=dict(prefill_gemm=pg,prefill_attention=pa,decode_gemm=dg,decode_attention=da)
    return dict(prefill=add(pg,pa),decode=add(dg,da),total=add(pg,pa,dg,da),components=components)


def execute_reference(x,w,groups,columns):
    """Executable FP32 partial-sum schedule, including K/N tails; no timing claim."""
    import numpy as np
    assert x.ndim==1 and w.shape[0]==len(x)
    k,n=w.shape;part=base.ceildiv(k,groups);out=np.zeros(n,dtype=np.float32)
    for j in range(0,n,columns):
        partials=[]
        for g in range(groups):
            first=min(g*part,k);last=min((g+1)*part,k)
            partials.append(np.asarray(x[first:last]@w[first:last,j:j+columns],dtype=np.float32))
        while len(partials)>1:
            partials=[partials[i]+partials[i+1] if i+1<len(partials) else partials[i] for i in range(0,len(partials),2)]
        out[j:j+columns]=partials[0]
    return out
