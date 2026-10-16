"""Finite-credit weight-supply experiment, not an NPU/DRAM simulator.

N ownership is fixed. Equal numbers/rates of decoders are either bank-local or
pooled. The paid pool routes compressed input and expanded output; an ideal
zero-fabric pool is a diagnostic. No activation, accumulator, row timing, energy
or synthesized area is predicted. Service parameters are explicit scenarios.
"""
from collections import deque
from dataclasses import asdict, dataclass
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    bank_bytes_cycle: float = 12.8
    macs_per_bank: int = 64
    m: int = 4
    decode_lanes: int = 16
    escape_lanes: int = 1
    decoder_input_bytes_cycle: float = 32
    input_credits: int = 2
    output_credits: int = 2
    group: int = 1
    link_bytes_cycle: float = math.inf
    link_start_cycles: float = 0
    descriptor_bytes: int = 8
    # One 8-byte offset per block, read once and retained to issue that block.
    # Streams also read the finite table header/padding, distributed by bank.
    index_header_bytes: int = 64


def decode_cycles(block, cfg):
    stored, valid, escapes, mode = block
    if mode not in (0, 1) or stored < valid or valid < 8 or not 0 <= escapes <= 256:
        raise ValueError('Malformed block metadata')
    return max(math.ceil(256 / cfg.decode_lanes),
               math.ceil(valid / cfg.decoder_input_bytes_cycle),
               math.ceil(escapes / cfg.escape_lanes) if mode else 0)


def simulate(bank_blocks, cfg):
    """Event-driven finite buffers, in-order consumers and deterministic arbiter.

    A decoder is reserved while its compressed transfer is in flight. It is
    freed at decode completion; an output credit retains the expanded block
    until its owner consumes it. Input credits span issue through decode end.
    Separate pooled input/output links serialize transfers. Local and pooled
    paths both have the same decoder input/output widths and credit counts.
    """
    banks = len(bank_blocks)
    if not banks or cfg.group < 1 or banks % cfg.group or any(not b for b in bank_blocks):
        raise ValueError('Groups must partition the banks')
    if min(cfg.bank_bytes_cycle, cfg.macs_per_bank, cfg.m, cfg.decode_lanes,
           cfg.escape_lanes, cfg.decoder_input_bytes_cycle, cfg.input_credits,
           cfg.output_credits, cfg.link_bytes_cycle) <= 0 or cfg.link_start_cycles < 0:
        raise ValueError('Positive rates/counts and nonnegative link latency required')
    units = banks  # Fixed total decoder count in all organizations.
    available = [True] * units
    input_used = [0] * banks
    output_used = [0] * banks
    peak_input = [0] * banks
    peak_output = [0] * banks
    read_busy = [False] * banks
    compute_busy = [False] * banks
    next_read = [0] * banks
    next_compute = [0] * banks
    ready = [deque() for _ in range(banks)]
    decoded = [{} for _ in range(banks)]
    groups = banks // cfg.group
    input_link = [0.0] * groups
    output_link = [0.0] * groups
    read_work = [0.0] * banks
    decode_work = [0.0] * banks
    compute_work = [0.0] * banks
    jobs = sum(map(len, bank_blocks))
    events = []
    serial = 0
    done = 0
    transfers_in = transfers_out = 0
    totals = dict(decoder_wait_cycles=0.0, input_backpressure_cycles=0.0,
                  output_backpressure_cycles=0.0, owner_reorder_wait_cycles=0.0)
    timestamp = 0.0

    def event(t, kind, bank, index, unit=-1):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (t, serial, kind, bank, index, unit))

    def transfer(t, size, links, group):
        if cfg.group == 1 or math.isinf(cfg.link_bytes_cycle):
            return t
        start = max(t, links[group])
        # Pipelined link: propagation latency does not occupy the serialization
        # resource. Charging it per packet as throughput would weaken the pool.
        links[group] = start + size / cfg.link_bytes_cycle
        return links[group] + cfg.link_start_cycles

    while done < jobs:
        # Start consumer only in original per-bank block order.
        for b in range(banks):
            i = next_compute[b]
            if not compute_busy[b] and i in decoded[b]:
                totals['owner_reorder_wait_cycles'] += timestamp - decoded[b].pop(i)
                compute_busy[b] = True
                duration = math.ceil(256 * cfg.m / cfg.macs_per_bank)
                compute_work[b] += duration
                event(timestamp + duration, 'consume', b, i)
        # Ready-first deterministic arbitration among all available units.
        for unit in range(units):
            if not available[unit]:
                continue
            group = unit // cfg.group
            lo, hi = group * cfg.group, (group + 1) * cfg.group
            candidates = [(ready[b][0][0], b) for b in range(lo, hi)
                          if ready[b] and output_used[b] < cfg.output_credits]
            if not candidates:
                continue
            _, b = min(candidates)
            arrival, i = ready[b].popleft()
            totals['decoder_wait_cycles'] += timestamp - arrival
            output_used[b] += 1
            peak_output[b] = max(peak_output[b], output_used[b])
            available[unit] = False
            block = bank_blocks[b][i]
            start = transfer(timestamp, block[0]+cfg.descriptor_bytes, input_link, group)
            duration = decode_cycles(block, cfg)
            decode_work[b] += duration
            if cfg.group > 1:
                transfers_in += block[0]+cfg.descriptor_bytes
            event(start + duration, 'decode', b, i, unit)
        # Bank service includes padded blocks plus amortized offset/header read.
        for b in range(banks):
            if read_busy[b] or next_read[b] == len(bank_blocks[b]) or input_used[b] == cfg.input_credits:
                continue
            i = next_read[b]
            next_read[b] += 1
            input_used[b] += 1
            peak_input[b] = max(peak_input[b], input_used[b])
            read_busy[b] = True
            size = bank_blocks[b][i][0] + 8
            if i == 0:
                size += cfg.index_header_bytes / banks
            duration = size / cfg.bank_bytes_cycle
            read_work[b] += duration
            event(timestamp + duration, 'read', b, i)
        if not events:
            raise RuntimeError('Pipeline deadlock')
        target = events[0][0]
        dt = target - timestamp
        for b in range(banks):
            if not read_busy[b] and next_read[b] < len(bank_blocks[b]) and input_used[b] == cfg.input_credits:
                totals['input_backpressure_cycles'] += dt
            if ready[b] and output_used[b] == cfg.output_credits:
                totals['output_backpressure_cycles'] += dt
        timestamp = target
        # Process all simultaneous completions before dispatching again.
        while events and events[0][0] == timestamp:
            _, _, kind, b, i, unit = heapq.heappop(events)
            if kind == 'read':
                read_busy[b] = False
                ready[b].append((timestamp, i))
            elif kind == 'decode':
                available[unit] = True
                input_used[b] -= 1
                arrive = transfer(timestamp, 512+cfg.descriptor_bytes, output_link, b // cfg.group)
                if cfg.group > 1:
                    transfers_out += 512+cfg.descriptor_bytes
                event(arrive, 'arrive', b, i)
            elif kind == 'arrive':
                decoded[b][i] = timestamp
            elif kind == 'consume':
                compute_busy[b] = False
                next_compute[b] += 1
                output_used[b] -= 1
                done += 1
        if any(i < 0 or i > cfg.input_credits for i in input_used) or any(o < 0 or o > cfg.output_credits for o in output_used):
            raise RuntimeError('Credit conservation failed')
    if events or any(input_used) or any(output_used) or next_compute != list(map(len, bank_blocks)):
        raise RuntimeError('Lost/duplicated block')
    payload = sum(block[0] for bank in bank_blocks for block in bank)
    return dict(cycles=timestamp, blocks=done, weight_consumer_macs=done*256*cfg.m,
                decoded_weight_bytes=done*512, bank_bytes=payload+8*jobs+cfg.index_header_bytes,
                crossbar_input_bytes=transfers_in, crossbar_output_bytes=transfers_out,
                max_input_credits=max(peak_input), max_output_credits=max(peak_output),
                # Support the codec's 576-byte raw escape even when a sample
                # happens to have no such block. Do not right-size to the data.
                input_fifo_provision_bytes=banks*cfg.input_credits*(576+cfg.descriptor_bytes),
                output_fifo_provision_bytes=banks*cfg.output_credits*(512+cfg.descriptor_bytes),
                max_bank_service_cycles=max(read_work),
                decoder_work_cycles=sum(decode_work),
                decoder_load_balance_ceiling=max(decode_work)*banks/sum(decode_work),
                max_consumer_service_cycles=max(compute_work), **totals)


def main():
    root = Path(__file__).resolve().parents[1]
    inp = root/'evidence/architecture/supply_blocks.json'
    source = json.loads(inp.read_text())
    results = []
    for model in source['models']:
        for row in model['rows']:
            blocks = [b['blocks'] for b in row['banks']]
            for bandwidth, macs in [(6.4, 128), (12.8, 64), (25.6, 64), (51.2, 32)]:
                for m in (1, 4, 8, 29):
                    common = dict(bank_bytes_cycle=bandwidth, macs_per_bank=macs, m=m,
                                  index_header_bytes=row['address_table_bytes']-8*sum(map(len, blocks)))
                    variants = {}
                    configs = [('local', {}), ('ideal_pool', dict(group=16))]
                    for group in (4, 16):
                        for factor in (.5, 1, 2):
                            configs.append((f'routed_g{group}_x{factor:g}',
                                            dict(group=group, link_bytes_cycle=group*32*factor,
                                                 link_start_cycles=2)))
                    for label, kwargs in configs:
                        variants[label] = simulate(blocks, Config(**common, **kwargs))
                    results.append(dict(model=model['model'], tensor=row['tensor'], m=m,
                                        bank_gb_s=bandwidth, macs=macs, variants=variants))
    result = dict(scope=__doc__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),
                  default_config={k:v for k,v in asdict(Config()).items() if k != 'link_bytes_cycle'},
                  routed_pool='Groups 4/16. Separate input/output links, each group*32*{0.5,1,2} B/cycle, 2 cycles pipelined propagation. Sum link widths 256/512/1024 B/cycle per direction. Ports/bytes/buffers charged, no fabricated area or energy.',
                  bank_parameters='1 GHz, per-PE peak HB bandwidth/MAC counts anchored to H2-LLM Table 6 at 40 nm; effective bank timing and extra decoder area NOT calibrated.',
                  results=results)
    result['escape_controls'] = []
    # Target the strongest routed observations, not a newly selected workload.
    # Extra local escape lanes cost logic too: this is a capability control,
    # explicitly not an iso-area comparison or a claimed cheaper design.
    for entry in results:
        if entry['variants']['local']['cycles']/entry['variants']['routed_g16_x2']['cycles'] <= 1.05:
            continue
        model = next(m for m in source['models'] if m['model'] == entry['model'])
        row = next(r for r in model['rows'] if r['tensor'] == entry['tensor'])
        blocks = [b['blocks'] for b in row['banks']]
        controls = []
        for lanes in (1, 2, 4):
            for group in (1, 16):
                cfg = Config(bank_bytes_cycle=entry['bank_gb_s'], macs_per_bank=entry['macs'],
                             m=entry['m'], escape_lanes=lanes, group=group,
                             link_bytes_cycle=1024, link_start_cycles=2)
                controls.append(dict(escape_lanes=lanes, group=group, **simulate(blocks, cfg)))
        result['escape_controls'].append({k:v for k,v in entry.items() if k != 'variants'} | dict(controls=controls))
    speedups = {label:[r['variants']['local']['cycles']/r['variants'][label]['cycles'] for r in results]
                for label in results[0]['variants']}
    summary = {label:dict(min=min(v), max=max(v), median=(sorted(v)[len(v)//2-1]+sorted(v)[len(v)//2])/2,
                         points=len(v), above_1pct=sum(x>1.01 for x in v), above_5pct=sum(x>1.05 for x in v))
               for label,v in speedups.items()}
    result['summary'] = summary
    out = root/'evidence/architecture/bank_supply.json.gz'
    out.write_bytes(gzip.compress((json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n').encode(), mtime=0))
    (root/'evidence/architecture/bank_supply_summary.json').write_text(json.dumps(dict(
        scope='Unweighted finite-sample case counts and ratios, NOT full-model/agent performance',
        source_sha256=result['source_sha256'], input_sha256=result['input_sha256'], summary=summary,
        escape_controls=result['escape_controls']), indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
