"""One fixed K2 organization extended to M=1..8, with explicit reuse/costs.

No extra MACs, banks, HB capacity or interface rate. Both the strong unified
baseline and the new local path get this mapping. Larger batches use the
ordinary mapping, never get split merely to help the candidate.
"""
from dataclasses import replace
from functools import lru_cache
import math

from architecture import model as base, mechanism as mech, supply_path as supply
from architecture.family_replay import batch_capacity, segments


def provision(hw,spec):
    # Existing g=2 address generators gain a row counter/sequencer. Reduction
    # uses the same lanes repeatedly; partial SRAM is taken from existing SRAM.
    controllers=hw.arrays*2
    area=controllers*mech.KShard().address_mm2
    clock=1/1.005
    power=controllers*.01  # explicit uncalibrated controller dynamic envelope
    h=replace(hw,name=hw.name+'_m8',frequency=hw.frequency*clock,
        sram_bandwidth_per_array=hw.sram_bandwidth_per_array*clock,
        area_mm2=hw.area_mm2+area,cost_units=hw.cost_units+area*spec['logic_assumptions']['cost_per_mm2'],
        base_power_w=hw.base_power_w+power+area*.1,peak_power_w=hw.peak_power_w+power+area*.1)
    bad=[k for k,v in [('cost_units',h.cost_units),('area_mm2',h.area_mm2),('power_w',h.peak_power_w),
                       ('shoreline_mm',h.shoreline_mm)] if v>spec['budget'][k]]
    return h,dict(row_controllers=controllers,added_area_mm2=area,added_peak_w=power+area*.1,
                  clock_factor=clock,max_partial_buffer_bytes=hw.arrays*2*4*2*8*hw.cols),bad


def execute_reference(x,w,columns):
    """Executable two-part K reduction; FP32 partials, M and N tails retained."""
    import numpy as np
    m,k=x.shape;n=w.shape[1];part=base.ceildiv(k,2)
    result=np.zeros((m,n),dtype=np.float32)
    for j in range(0,n,columns):
        partials=[]
        for group in range(2):
            begin=group*part;end=min(k,begin+part)
            partials.append(x[:,begin:end].astype(np.float32)@w[begin:end,j:j+columns].astype(np.float32))
        result[:,j:j+columns]=partials[0]+partials[1]
    return result


def sharded(hw,m,k,n,path=None,fraction=0):
    assert 1<=m<=8 and hw.rows%2==0
    g=2;c=hw.cols;subrows=hw.rows//g;kp=base.ceildiv(k,g)
    tiles=base.ceildiv(n,c);waves=base.ceildiv(tiles,hw.arrays)
    reserved=2*4*g*m*c
    local=(hw.sram_mib*base.MIB-supply.dma_buffer_bytes(hw))/hw.arrays
    legal=[]
    for t in (128,512):
        tk=min(kp,t)
        shared_footprint=2*(2*g*m*tk+2*g*tk*c)+reserved
        # Do not route a weight tile through a FIFO too small to preserve it
        # for all m rows. Shared input and partials stay in the charged SRAM.
        hb_half=path.buffer_bytes/(2*hw.arrays) if path else math.inf
        if shared_footprint<=local and fraction*2*g*tk*c<=hb_half:
            legal.append(tk)
    if not legal:return None
    tk=max(legal)
    weight=2*k*n;activation=2*m*k*tiles;writes=2*m*n
    cycles=waves*(kp+subrows+c-2)
    scheme=mech.KShard(groups=2)
    red_bytes=4*m*n*(2*g-2)
    reduction_sram=4*g*m*n+4*m*n
    tree_cycles=waves*(base.ceildiv(m*c,scheme.reduction_lanes)+
                       math.ceil(math.log2(g))*scheme.adder_latency_cycles+scheme.sync_cycles)
    reduction=max(tree_cycles/hw.frequency,
        red_bytes/(hw.arrays*scheme.reduction_link_bytes_cycle*hw.frequency),
        reduction_sram/(hw.arrays*hw.sram_bandwidth_per_array))
    row=dict(compute_seconds=cycles/hw.frequency,flops=2*m*k*n,
        read_bytes=weight+activation,write_bytes=writes,
        weight_read_bytes=weight,kv_read_bytes=0,activation_read_bytes=activation,
        partial_read_bytes=0,activation_write_bytes=writes,partial_write_bytes=0,
        sram_bytes=weight+activation+4*g*m*n+reduction_sram,
        reduction_seconds=reduction,reduction_adds=(g-1)*m*n,
        reduction_link_bytes=red_bytes,
        # X fanout retains the old conservative accounting; forwarding W
        # through additional live rows is charged to BOTH hardware designs.
        input_distribution_bytes=2*m*k*n+(m-1)*weight,
        output_columns=m*n,k_groups=g,counted_k_device=True,dataflow='K2_SMALL_M',
        tile=[m,tk,c],buffer_bytes=2*(2*g*m*tk+2*g*tk*c)+reserved)
    result=supply.score(hw,row,path,fraction,allow_hb=path is not None)
    result.update(partial_buffer_bytes=reserved,shared_tile_bytes=row['buffer_bytes'],
                  hb_weight_tile_bytes=fraction*2*g*tk*c,
                  hb_half_fifo_per_array=path.buffer_bytes/(2*hw.arrays) if path else 0,
                  reuse_m=m)
    return result


@lru_cache(maxsize=100000)
def linear(hw,m,k,n,mode,path=None,fraction=0):
    if mode=='old_m1':return supply.linear(hw,m,k,n,True,path,fraction)
    assert mode in ('baseline','small_m')
    normal=supply.linear(hw,m,k,n,True)  # legal ordinary mapping / old K2
    if m>8:return normal
    options=[normal]
    plain=sharded(hw,m,k,n)
    if plain:options.append(plain)
    if mode=='small_m':
        local=sharded(hw,m,k,n,path,fraction)
        if local:options.append(local)
    return min(options,key=lambda r:(r['seconds'],r['energy_j']))


@lru_cache(maxsize=10000)
def attention(hw,length,heads,kv_heads,dim,batch,decode):
    return supply.score(hw,mech.attention(supply.tile_hardware(hw),length,heads,kv_heads,dim,batch,decode))


def forward(hw,cfg,length,batch,decode,mode,path=None,fraction=0,embedding=False):
    h,d=cfg['hidden_size'],cfg['head_dim'];q=cfg['num_attention_heads']*d;kv=cfg['num_key_value_heads']*d
    ff,layers=cfg['intermediate_size'],cfg['num_hidden_layers'];m=batch if decode else batch*length
    dense=supply.scale(supply.add(*(linear(hw,m,k,n,mode,path,fraction) for k,n in
                       [(h,q),(h,kv),(h,kv),(q,h),(h,ff),(h,ff),(ff,h)])),layers)
    if not embedding:dense=supply.add(dense,linear(hw,batch,h,cfg['vocab_size'],mode,path,fraction))
    return supply.add(dense,supply.scale(attention(hw,length,cfg['num_attention_heads'],
        cfg['num_key_value_heads'],d,batch,decode),layers))


def execute_batch(hw,cfg,calls,mode,path=None,fraction=0,resident_weight_bytes=None):
    capacity=batch_capacity(cfg,calls,resident_weight_bytes)
    if capacity>hw.capacity_bytes:raise ValueError(f'Batch capacity {capacity}>{hw.capacity_bytes}')
    pre=forward(hw,cfg,max(c['L'] for c in calls),len(calls),False,mode,path,fraction)
    clock=pre['seconds'];finished={c['id']:clock for c in calls if c['O']==1}
    rows=[]
    for p in segments(calls):
        # Exact step count and independent L/O. Midpoint attention matches the
        # frozen analytical convention; it is not an exact cycle simulation.
        r=supply.scale(forward(hw,cfg,p['padded_prompt']+(p['start']+p['end']+1)//2,
            p['batch'],True,mode,path,fraction),p['count'])
        clock+=r['seconds']
        rows.append(dict(**p,service=r))
        finished.update({c['id']:clock for c in calls if c['O']-1==p['end']})
    assert set(finished)=={c['id'] for c in calls}
    dec=supply.add(*(r['service'] for r in rows))
    return dict(prefill=pre,decode=dec,total=supply.add(pre,dec),segments=rows,
                completion_seconds=finished,peak_capacity_bytes=capacity)
