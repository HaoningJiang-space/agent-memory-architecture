"""Conditional bank/interface/buffer model over frozen observed requests.

This is a supply-path experiment, not a new HB chip PPA claim. The original
model and KShard remain unchanged. A static HB copy serves only M=1 weights;
all other traffic and uncached weights use the original capacity memory.
"""
from dataclasses import dataclass, asdict, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics

from architecture import model as base
from architecture import mechanism as mech
from architecture.explore import load
from architecture.joint_diagnostics import workloads

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/architecture'
GIB = 2**30
# H2-LLM Sec. 7.1, Table 6: fractions of one 6.76 mm2 PE at 40 nm.
HB_CONTROLLER_FRACTION = {6.4: .046, 12.8: .107, 25.6: .197, 51.2: .402}
LIMITS = ('compute', 'backing', 'hb_bank', 'hb_interface', 'hb_window', 'hb_sram', 'shared_sram')
FIELDS = ('seconds', 'energy_j', 'flops', 'weight_read_bytes', 'kv_read_bytes',
          'activation_read_bytes', 'partial_read_bytes', 'activation_write_bytes', 'partial_write_bytes',
          'backing_read_bytes', 'backing_write_bytes', 'hb_read_bytes',
          'shared_sram_bytes', 'shared_compute_bytes', 'dma_sram_bytes', 'local_sram_bytes', 'reduction_seconds',
          'backing_seconds', 'hb_seconds', 'compute_seconds', 'shared_sram_seconds') + tuple('limited_by_' + k + '_seconds' for k in LIMITS) + ('limited_by_reduction_seconds',)


@dataclass(frozen=True)
class Route:
    banks: int
    bank_bytes_s: float
    interface_bytes_s: float
    latency_s: float
    outstanding_per_bank: int
    payload_bytes: int
    bank_skew: float = 1.0
    interface_efficiency: float = .8

    def capacities(self):
        return dict(bank=self.banks * self.bank_bytes_s / self.bank_skew,
                    interface=self.interface_bytes_s * self.interface_efficiency,
                    window=self.banks * self.outstanding_per_bank * self.payload_bytes / self.latency_s)

    def service(self, size):
        if not size:
            return 0.0
        # A streaming lower-envelope approximation: finite in-flight window is
        # a throughput ceiling, plus one startup. No per-burst drain/restart.
        return self.latency_s + size / min(self.capacities().values())


@dataclass(frozen=True)
class LocalPath:
    banks: int = 32
    gb_s_per_bank: float = 51.2
    ports_per_bank: int = 4
    bytes_per_port_cycle: int = 16
    buffer_bytes_per_port: int = 4096
    outstanding_per_bank: int = 2
    capacity_bytes: int = 10_000_000_000
    bank_skew: float = 1.0
    controller_area_scale: float = 1.0
    incremental_memory_cost: float = 100.0
    extra_clock_loss: float = .01
    latency_ns: float = 100.0
    bank_efficiency: float = .8
    interface_efficiency: float = .8
    read_pj_bit: float = .88
    local_sram_pj_bit: float = .1
    # No SRAM compiler at the target node: preserve published 40nm density.
    sram_mm2_mb: float = 2.72
    input_wire_pj_bit: float = .2
    background_w_gb: float = .075

    @property
    def buffer_bytes(self):
        # Two independent single-port banks ping-pong, not a free dual port.
        return 2 * self.banks * self.ports_per_bank * self.buffer_bytes_per_port

    def route(self):
        return Route(self.banks, self.gb_s_per_bank * 1e9 * self.bank_efficiency,
                     self.banks * self.gb_s_per_bank * 1e9, self.latency_ns * 1e-9,
                     self.outstanding_per_bank, self.buffer_bytes_per_port,
                     self.bank_skew, self.interface_efficiency)

    def port_bytes_s(self, hw):
        # Producer and consumer use different halves; both require this rate.
        return self.banks * self.ports_per_bank * self.bytes_per_port_cycle * hw.frequency / self.bank_skew


def backing_route(hw, skew=1.0, payload=16384, outstanding=1):
    # 16 independently serviceable partitions per package is a declared model,
    # not the vendor's physical bank count. Baseline and candidate share it.
    banks = hw.packages * 16
    return Route(banks, hw.bandwidth * hw.bandwidth_efficiency / banks,
                 hw.bandwidth, hw.latency, outstanding, payload, skew, hw.bandwidth_efficiency)


def dma_buffer_bytes(hw):
    route = backing_route(hw)
    return route.banks * route.outstanding_per_bank * route.payload_bytes


def tile_hardware(hw):
    # DMA responses cannot use a buffer already fully occupied by operands.
    reserve = math.ceil(dma_buffer_bytes(hw) / base.MIB)
    assert hw.sram_mib > reserve
    return replace(hw, sram_mib=hw.sram_mib-reserve)


def provision(hw, path, spec):
    """Incremental costs over unchanged K2 provision, with unknown HB price."""
    controller = path.banks * 6.76 * HB_CONTROLLER_FRACTION[path.gb_s_per_bank] * path.controller_area_scale
    fifo = path.buffer_bytes / 1e6 * path.sram_mm2_mb
    # One address/queue controller per lane, explicit unsynthesized coefficient
    # inherited from KShard; storage and request slots already counted above.
    address = path.banks * path.ports_per_bank * .002
    area = controller + fifo + address
    clock = 1 / (1 + path.extra_clock_loss)
    memory_bg = path.capacity_bytes / 1e9 * path.background_w_gb
    static = area * .1 + memory_bg
    io_peak = path.banks * path.gb_s_per_bank * 1e9 * 8 * path.read_pj_bit * 1e-12
    # Full read AND write local-buffer activity + local input wires; conservative.
    fifo_peak = 2 * path.port_bytes_s(hw) * 8 * path.local_sram_pj_bit * 1e-12
    wire_peak = path.port_bytes_s(hw) * 8 * path.input_wire_pj_bit * 1e-12
    peak = hw.peak_power_w + static + io_peak + fifo_peak + wire_peak
    cost_without_hb_price = hw.cost_units + area * spec['logic_assumptions']['cost_per_mm2']
    cost = cost_without_hb_price + path.incremental_memory_cost
    result = replace(hw, name=hw.name + f'_hb{path.banks}b{path.gb_s_per_bank:g}p{path.ports_per_bank}',
                     frequency=hw.frequency * clock,
                     sram_bandwidth_per_array=hw.sram_bandwidth_per_array * clock,
                     area_mm2=hw.area_mm2 + area, cost_units=cost,
                     peak_power_w=peak, base_power_w=hw.base_power_w + static)
    charges = dict(controller_mm2=controller, local_sram_mm2=fifo, address_mm2=address,
                   buffer_bytes=path.buffer_bytes, local_sram_single_ports=2*path.banks*path.ports_per_bank,
                   interface_data_pins=path.banks * int(path.gb_s_per_bank * 8 / .4),
                   hb_capacity_bytes=path.capacity_bytes, incremental_memory_cost=path.incremental_memory_cost,
                   max_incremental_memory_cost=spec['budget']['cost_units'] - cost_without_hb_price,
                   added_power_envelope_w=peak - hw.peak_power_w, added_static_w=static,
                   area_mm2=result.area_mm2, cost_units=cost, power_w=peak,
                   shoreline_mm=result.shoreline_mm, clock_factor=clock,
                   footprint_note='450 mm2 is the existing logic-area budget, not package footprint. A 10 GB HybridSpec stack has a separate 408 mm2 footprint; this is NOT a layout of that stack.')
    rejected = [k for k in ('cost_units', 'area_mm2', 'power_w', 'shoreline_mm') if charges[k] > spec['budget'][k]]
    return result, charges, rejected


def add(*rows):
    return {k: sum(r.get(k, 0) for r in rows) for k in FIELDS}


def scale(row, count):
    return {k: v * count for k, v in add(row).items()}


def score(hw, row, path=None, fraction=0.0, allow_hb=False):
    """Reprice an identical mapping with explicit simultaneous supply domains.

The two DRAM domains can stream disjoint K/N weight partitions concurrently;
weights missing from the resident HB copy never use its nominal bandwidth.
SRAM partials, input fanout and K2's serialized reduction remain charged.
"""
    f = fraction if path is not None and allow_hb else 0.0
    hb_bytes = row.get('weight_read_bytes', 0) * f
    backing_reads = row['read_bytes'] - hb_bytes
    writes = row['write_bytes']
    dram = backing_route(hw)
    back = dram.service(backing_reads + writes)
    # KShard already includes serialized reduction SRAM reads/writes.
    reduction = row.get('reduction_seconds', 0)
    reduction_sram = (4*row.get('k_groups', 0)*row.get('output_columns', 0) +
                      4*row.get('output_columns', 0)) if reduction else 0
    shared_compute = max(0, row['sram_bytes'] - reduction_sram - hb_bytes)
    # Explicit shared-port accounting includes DMA fills and writeback reads;
    # the frozen model counted consumer accesses only. Apply to EVERY design.
    dma_sram = backing_reads + writes
    shared = shared_compute + dma_sram
    shared_time = shared / (hw.arrays * hw.sram_bandwidth_per_array)
    times = dict(compute=row['compute_seconds'], backing=back, shared_sram=shared_time)
    local = 0.0
    hb = 0.0
    if hb_bytes:
        capacities = path.route().capacities()
        times.update({'hb_' + k: path.route().latency_s + hb_bytes / v for k, v in capacities.items()})
        times['hb_sram'] = hb_bytes / path.port_bytes_s(hw)
        hb = max(times[k] for k in ('hb_bank', 'hb_interface', 'hb_window', 'hb_sram'))
        local = 2 * hb_bytes
    winner = max(times, key=times.get)
    elapsed = times[winner] + reduction
    # Dynamic components are recomputed, not rescaled API time. Existing tree,
    # mux and local input fanout remain exactly the frozen KShard energy terms.
    energy = (row['flops']/2*hw.mac_pj*1e-12 +
              8*(backing_reads*hw.read_pj_bit + writes*hw.write_pj_bit)*1e-12 +
              (shared + reduction_sram)*8*hw.sram_pj_bit*1e-12 + hw.static_w*elapsed)
    if path is not None:
        energy += 8*(hb_bytes*path.read_pj_bit + local*path.local_sram_pj_bit)*1e-12
        # Charge a local data path for every resident byte as a conservative
        # bound; no off-chip weight duplicate or activation has disappeared.
        energy += hb_bytes*8*path.input_wire_pj_bit*1e-12
    scheme = mech.KShard(groups=2)
    energy += (row.get('reduction_adds', 0)*scheme.adder_pj +
               8*(row.get('reduction_link_bytes', 0)+row.get('input_distribution_bytes', 0))*scheme.wire_pj_bit +
               row['flops']/2*scheme.mux_pj_mac if row.get('counted_k_device', False) else 0)*1e-12
    result = dict(seconds=elapsed, energy_j=energy, flops=row['flops'],
                  weight_read_bytes=row.get('weight_read_bytes', 0), kv_read_bytes=row.get('kv_read_bytes', 0),
                  activation_read_bytes=row.get('activation_read_bytes', 0),
                  partial_read_bytes=row.get('partial_read_bytes', 0),
                  activation_write_bytes=row.get('activation_write_bytes', 0),
                  partial_write_bytes=row.get('partial_write_bytes', 0),
                  backing_read_bytes=backing_reads, backing_write_bytes=writes, hb_read_bytes=hb_bytes,
                  shared_sram_bytes=shared+reduction_sram, shared_compute_bytes=shared_compute+reduction_sram,
                  dma_sram_bytes=dma_sram, local_sram_bytes=local,
                  reduction_seconds=reduction, backing_seconds=back,
                  hb_seconds=hb, compute_seconds=row['compute_seconds'], shared_sram_seconds=shared_time,
                  limited_by_reduction_seconds=reduction)
    result.update({'limited_by_'+k+'_seconds': times[winner] if k == winner else 0 for k in LIMITS})
    result.update(mapping=row.get('dataflow', 'attention'), fraction=f,
                  limiting=winner, resources=times, tile=row.get('tile'))
    return result


@lru_cache(maxsize=50000)
def linear(hw, m, k, n, counted=False, path=None, fraction=0.0):
    """Enumerate the frozen model's complete OS/WS tile set under new supply.

A fast device does not inherit a tile selected for another bandwidth. K2 is
an additional frozen mapping; no extra K-group design tuning is performed.
"""
    options = []
    local = (hw.sram_mib*base.MIB-dma_buffer_bytes(hw))/hw.arrays
    for tm in sorted({min(m, x) for x in (hw.rows,128,512,2048)}):
        for tn in sorted({min(n,x) for x in (hw.cols,512,2048)}):
            for tk in sorted({min(k,x) for x in (128,512)}):
                footprint = 2*(2*tm*tk+2*tk*tn)+4*tm*tn
                if footprint > local:
                    continue
                micro_m = base.ceildiv(m,tm)*base.ceildiv(tm,hw.rows)
                micro_n = base.ceildiv(n,tn)*base.ceildiv(tn,hw.cols)
                waves = base.ceildiv(micro_m*micro_n,hw.arrays)
                cycles = waves*(k+hw.rows+hw.cols-2)
                activation = 2*m*k*base.ceildiv(n,tn)
                weights = 2*k*n
                sram = 2*k*(m*base.ceildiv(n,hw.cols)+n*base.ceildiv(m,hw.rows))+4*m*n
                for flow in ('OS', 'WS'):
                    weight = weights*base.ceildiv(m,tm) if flow == 'OS' else weights
                    partial = 0 if flow == 'OS' else 4*m*n*(base.ceildiv(k,tk)-1)
                    row = dict(compute_seconds=cycles/hw.frequency, flops=2*m*k*n,
                               read_bytes=activation+weight+partial, write_bytes=2*m*n+partial,
                               weight_read_bytes=weight, kv_read_bytes=0, activation_read_bytes=activation,
                               partial_read_bytes=partial, activation_write_bytes=2*m*n,
                               partial_write_bytes=partial, sram_bytes=sram,
                               dataflow=flow, tile=[tm,tk,tn], counted_k_device=counted)
                    options.append(score(hw,row,path,fraction,allow_hb=m==1))
                    if path is not None and m == 1:
                        # The backing copy is retained: always allow bypass.
                        options.append(score(hw,row,path,0,allow_hb=False))
    if counted and m == 1:
        row = mech.sharded(tile_hardware(hw),mech.KShard(groups=2),m,k,n)
        if row is not None:
            row = dict(row,output_columns=n,counted_k_device=True)
            options.append(score(hw,row,path,fraction,allow_hb=True))
            if path is not None:
                options.append(score(hw,row,path,0,allow_hb=False))
    if not options:
        raise ValueError('No legal tile')
    return min(options, key=lambda r:(r['seconds'], r['energy_j']))


def attention(hw, config, length, decode, counted):
    row = mech.attention(tile_hardware(hw),length,config['num_attention_heads'],config['num_key_value_heads'],config['head_dim'],decode=decode)
    # Attention has no weight-routing benefit; no mux-MAC charge from K2 in the
    # original implementation, only the device's global frequency/leakage.
    return score(hw,row)


def evaluate(hw,config,L,O,counted=False,path=None,fraction=0.0):
    h,d=config['hidden_size'],config['head_dim']
    q,kv=config['num_attention_heads']*d,config['num_key_value_heads']*d
    ff,layers=config['intermediate_size'],config['num_hidden_layers']
    shapes=[(h,q),(h,kv),(h,kv),(q,h),(h,ff),(h,ff),(ff,h)]
    def dense(m):
        return scale(add(*(linear(hw,m,k,n,counted,path,fraction) for k,n in shapes)),layers)
    pre=add(dense(L),linear(hw,1,h,config['vocab_size'],counted,path,fraction),
            scale(attention(hw,config,L,False,counted),layers))
    steps=O-1
    dec=scale(add(dense(1),linear(hw,1,h,config['vocab_size'],counted,path,fraction),
                  scale(attention(hw,config,L+(steps+1)//2,True,counted),layers)),steps) if steps else add()
    return dict(prefill=pre,decode=dec,total=add(pre,dec))


def fraction_options(weight_bytes,path):
    maximum=min(1,path.capacity_bytes/weight_bytes) if path else 0.0
    return sorted(set((0.0,maximum*.25,maximum*.5,maximum*.75,maximum)))


def cold_fill(hw,path,weight_bytes,fraction):
    size=weight_bytes*fraction
    if not size:return dict(bytes=0,seconds=0,energy_j=0)
    # Conservative read/write energy: write uses 1.2x declared HB read anchor.
    seconds=max(backing_route(hw).service(size),path.route().service(size),size/path.port_bytes_s(hw))
    energy=size*8*(hw.read_pj_bit+1.2*path.read_pj_bit+2*path.local_sram_pj_bit+path.input_wire_pj_bit)*1e-12+hw.static_w*seconds
    return dict(bytes=size,seconds=seconds,energy_j=energy)


@lru_cache(maxsize=10000)
def capacity_requirement(config_json,L,O,weight_bytes):
    config=json.loads(config_json)
    shape=base.decoder_shape(config)
    kv=shape['kv_bytes_per_token']*(L+max(0,O-1))
    # Logical activation buffers from the frozen model, no CUDA reservation.
    activation=2*L*(config['hidden_size']+2*config['intermediate_size'])
    return dict(weights_bytes=weight_bytes,kv_bytes=kv,activation_bytes=activation,total_bytes=weight_bytes+kv+activation)


def profile(hw,calls,trace,weights,counted=False,path=None,fixed_fraction=None):
    # One static placement fraction per model/path, not per request or layer.
    weight_bytes=weights[calls[0]['model']]
    capacity_rows=[capacity_requirement(json.dumps(trace['models'][c['model']]['config'],sort_keys=True),c['L'],c['O'],weight_bytes) for c in calls]
    capacity_peak=max(capacity_rows,key=lambda r:r['total_bytes'])
    assert capacity_peak['total_bytes'] <= hw.capacity_bytes
    choices=[]
    for fraction in (fraction_options(weight_bytes,path) if fixed_fraction is None else [fixed_fraction]):
        assert 0 <= fraction <= (min(1,path.capacity_bytes/weight_bytes) if path else 0) + 1e-12
        rows=[evaluate(hw,trace['models'][c['model']]['config'],c['L'],c['O'],counted,path,fraction) for c in calls]
        mean={phase:{k:statistics.mean(r[phase][k] for r in rows) for k in FIELDS} for phase in ('total','prefill','decode')}
        choices.append(dict(fraction=fraction,mean=mean))
    selected=min(choices,key=lambda c:(c['mean']['total']['seconds'],c['mean']['total']['energy_j']))
    selected['capacity_fraction_max']=min(1,path.capacity_bytes/weight_bytes) if path else 0
    selected['capacity_requirement']=capacity_peak
    selected['backing_capacity_bytes']=hw.capacity_bytes
    selected['dma_buffer_bytes']=dma_buffer_bytes(hw)
    selected['model_weight_bytes']=weight_bytes
    selected['placement_weight_bytes']=weight_bytes*selected['fraction']
    selected['fraction_choices']=[dict(fraction=c['fraction'],seconds=c['mean']['total']['seconds'],energy_j=c['mean']['total']['energy_j']) for c in choices]
    selected['cold_fill']=cold_fill(hw,path,weight_bytes,selected['fraction']) if path else dict(bytes=0,seconds=0,energy_j=0)
    selected['dominant']=max(LIMITS,key=lambda k:selected['mean']['total']['limited_by_'+k+'_seconds'])
    selected['requests']=[dict(id=c['id'],model=c['model'],L=c['L'],O=c['O']) for c in calls]
    return selected


def main():
    spec,trace,_,capacity,receipts=load()
    original,_=base.candidates(spec)
    weights={m['model']:m['weight_bytes'] for m in capacity['models']}
    work={k:v for k,v in workloads(trace).items() if k in ('update','raw_raw','mem0_mem0')}
    designs=[];profiles=[];rejected=[];headroom=[]
    # Expose three unknown memory/assembly prices. Controller and FIFO costs
    # remain explicit; these are conditional feasibility slices, not quotes.
    prices=(40.,100.,200.)
    path_shapes=((32,25.6,2),(32,51.2,4),(64,25.6,2),(64,51.2,4))
    for original_hw in original:
        hw=replace(original_hw,fill_mode='continuous_k')
        designs.append(dict(name=hw.name,kind='unified_continuous_k',hardware=asdict(hw),path=None))
        for group,calls in work.items():
            profiles.append(dict(name=hw.name,kind='unified_continuous_k',group=group,price=None,**profile(hw,calls,trace,weights)))
        if hw.rows != 128:continue
        khw,kcharges,kbad=mech.provision(hw,mech.KShard(groups=2),spec)
        assert not kbad
        designs.append(dict(name=khw.name,kind='frozen_k2',hardware=asdict(khw),path=None,kcharges=kcharges))
        for group,calls in work.items():
            profiles.append(dict(name=khw.name,kind='frozen_k2',group=group,price=None,**profile(khw,calls,trace,weights,True)))
        for banks,rate,ports in path_shapes:
            prototype=LocalPath(banks=banks,gb_s_per_bank=rate,ports_per_bank=ports,incremental_memory_cost=0)
            proto_hw,proto_charge,proto_bad=provision(khw,prototype,spec)
            headroom.append(dict(base=khw.name,path=asdict(prototype),charges=proto_charge,violations_without_memory_price=proto_bad))
            for price in prices:
                path=replace(prototype,incremental_memory_cost=price)
                shw,charges,bad=provision(khw,path,spec)
                name=shw.name+f'_price{price:g}'
                if bad:
                    rejected.append(dict(name=name,violations=bad,charges=charges));continue
                # Price does not change physics: reuse cached hardware keys.
                phw=replace(shw,name=proto_hw.name,cost_units=proto_hw.cost_units)
                designs.append(dict(name=name,kind='hb_local',price=price,hardware=asdict(shw),path=asdict(path),charges=charges))
                for group,calls in work.items():
                    profiles.append(dict(name=name,kind='hb_local',group=group,price=price,
                                         **profile(phw,calls,trace,weights,True,replace(path,incremental_memory_cost=0))))
        print('supply',hw.name,flush=True)
    best=[]
    for group in work:
        base_rows=[p for p in profiles if p['group']==group and p['kind']!='hb_local']
        for metric in ('seconds','energy_j'):
            baseline=min(base_rows,key=lambda p:p['mean']['total'][metric])
            best.append(dict(group=group,kind='strong_baseline',price=None,metric=metric,name=baseline['name'],value=baseline['mean']['total'][metric]))
            for price in prices:
                rows=[p for p in profiles if p['group']==group and p['kind']=='hb_local' and p['price']==price]
                if not rows:continue
                candidate=min(rows,key=lambda p:p['mean']['total'][metric])
                best.append(dict(group=group,kind='hb_local',price=price,metric=metric,name=candidate['name'],value=candidate['mean']['total'][metric],
                                 relative=baseline['mean']['total'][metric]/candidate['mean']['total'][metric]))
    # Fixed hardware, fixed placement: do not reselect a winner after applying
    # physical uncertainty. Include an empty-cache/no-resident counterexample.
    controls=[]
    for group in work:
        nominal=next(b for b in best if b['group']==group and b['kind']=='hb_local' and b['price']==100 and b['metric']=='seconds')
        design=next(d for d in designs if d['name']==nominal['name'])
        before=next(d for d in designs if d['name']==design['name'].split('_hb')[0])
        khw=base.Hardware(**before['hardware']);p=LocalPath(**design['path'])
        fixed_fraction=next(r for r in profiles if r['name']==nominal['name'] and r['group']==group)['fraction']
        for tag,new in [('nominal',p),('bank_skew2',replace(p,bank_skew=2)),('bank_skew4',replace(p,bank_skew=4)),
                        ('half_interface',replace(p,interface_efficiency=.4)),('single_port',replace(p,ports_per_bank=1)),('payload256',replace(p,buffer_bytes_per_port=256)),
                        ('clock_loss5pct',replace(p,extra_clock_loss=.05)),('controller_area2x',replace(p,controller_area_scale=2)),
                        ('no_resident_weights',p)]:
            shw,charges,bad=provision(khw,new,spec)
            controls.append(dict(group=group,tag=tag,charges=charges,violations=bad,
                                 **profile(shw,work[group],trace,weights,True,new,0.0 if tag=='no_resident_weights' else fixed_fraction)))
    samples=[]
    for group in ('update','mem0_mem0'):
        winner=next(b for b in best if b['group']==group and b['kind']=='hb_local' and b['price']==100 and b['metric']=='seconds')
        d=next(d for d in designs if d['name']==winner['name']);hw=base.Hardware(**d['hardware']);path=LocalPath(**d['path'])
        chosen=next(p for p in profiles if p['name']==winner['name'] and p['group']==group)
        model=work[group][0]['model'];cfg=trace['models'][model]['config'];h=cfg['hidden_size'];ff=cfg['intermediate_size']
        for tag,k,n in [('Q',h,cfg['num_attention_heads']*cfg['head_dim']),('up',h,ff),('down',ff,h)]:
            row=linear(hw,1,k,n,True,path,chosen['fraction'])
            samples.append(dict(group=group,model=model,operator=tag,shape=[1,k,n],design=winner['name'],
                result=row,required_hb_bytes_per_bank_compute_cycle=row['hb_read_bytes']/row['compute_seconds']/path.banks/hw.frequency,
                actual_bank_bytes_per_compute_cycle=path.route().capacities()['bank']/path.banks/hw.frequency,
                actual_local_read_port_bytes_per_cycle=path.ports_per_bank*path.bytes_per_port_cycle))
    paths=['architecture/supply_path.py','architecture/mechanism.py','evidence/architecture/finite_prefix.json']
    data=dict(scope=__doc__,budget=spec['budget'],input_sha256={**receipts,**{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}},
              workloads=work,designs=designs,profiles=profiles,rejected=rejected,headroom=headroom,best=best,controls=controls,
              weights=weights,prices=list(prices),path_shapes=path_shapes,operator_samples=samples,
              provenance=dict(h2_llm='Local ISCA 2025 PDF, Sec. 7.1/Table 6: HB rates 6.4/12.8/25.6/51.2 GB/s per bank; controller occupies 4.6/10.7/19.7/40.2% of 6.76 mm2 PE at 40nm; 2.72 mm2/MB SRAM; 0.88 pJ/bit HB I/O.',
                              hybridspec='Local ISCA 2026 PDF, Sec. VI.A: 10 GB HB, 4 TB/s, 408 mm2 stack footprint, 0.88 pJ/bit. We use only 10 GB as a capacity scenario, NOT that chip as our physical implementation.',
                              missing='Mixed technology anchors do not calibrate PPA: controller integration, SRAM bank timing, total die footprint, HB DRAM cost, floorplan, clock and network need independent validation.'),
              limitations=['Standalone independent requests; no batching or multi-agent end-to-end throughput claim.',
                           'Same finite L/O traces across all hardware, no algorithm quality comparison.',
                           'Uniform static weight partition; fractional placement approximates aligned tensor slices, not an implemented memory allocator.',
                           'HB is a duplicate persistent copy for M=1; full model and KV remain in backing memory. Every operator can bypass HB. Cold fill is separate and explicit.',
                           'Shared and local DRAM streams overlap optimistically, reduction remains serialized; no cycle-accurate bank or NoC model.',
                           'Attention, vector kernels, softmax and runtime omissions remain as in frozen architecture model.',
                           'Cost 40/100/200 are explicitly uncalibrated 10 GB memory/assembly prices in existing normalized units, not market prices.',
                           '450 mm2 logic-area budget does not bound package footprint; no physical fit assertion.'])
    (OUT/'supply_exploration.json').write_text(json.dumps(data,indent=2)+'\n')
    for result in best:print(result,flush=True)


if __name__=='__main__':main()
