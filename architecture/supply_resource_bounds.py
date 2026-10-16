"""Coupled HB-interface/MAC ceilings from local H2-LLM PDF Table 6.

Paper's 40 nm, 1 GHz, 6.76 mm2 PE is a reference, not our synthesized design.
No linear technology scaling or mixing with the central 10 nm area model.
The largest fitting FPU count excludes our additional decoder/network cost.
"""
import hashlib
import json
import math
from pathlib import Path


def max_m(macs, bandwidth, ratio, frequency=1e9):
    """BF16 compressed-weight-only intensity M/r versus 2*MAC*f/B."""
    if min(macs, bandwidth, ratio, frequency) <= 0:
        raise ValueError('Positive physical parameters required')
    return ratio * 2 * macs * frequency / bandwidth


def main():
    root = Path(__file__).resolve().parents[1]
    inputs = {name: root/'evidence/architecture'/name for name in
              ('hetero_projection_shapes.json', 'hetero_boundary.json', 'supply_blocks.json')}
    shapes = json.loads(inputs['hetero_projection_shapes.json'].read_text())
    old = json.loads(inputs['hetero_boundary.json'].read_text())
    source = json.loads(inputs['supply_blocks.json'].read_text())
    reference = []
    for bandwidth, fpus, area_fraction in [(6.4,8,.046), (12.8,4,.107), (25.6,4,.197), (51.2,2,.402)]:
        macs = fpus*16
        row = dict(hb_GB_s=bandwidth, macs=macs, fpu_area_mm2=fpus*.77,
                   controller_area_mm2=6.76*area_fraction, pe_area_mm2=6.76,
                   peak_GFLOPs=2*macs, encoded_weight_flop_per_byte=2*macs/bandwidth,
                   m_limit_raw=max_m(macs, bandwidth*1e9, 1),
                   m_limit_ratio_08=max_m(macs, bandwidth*1e9, .8), streams=[])
        for s in shapes['streams']:
            weight = sum(2*r['k']*r['n']*r['count'] for r in s['rows'])
            flops = sum(2*r['m']*r['k']*r['n']*r['count'] for r in s['rows'])
            eligible = [r for r in s['rows'] if r['m'] <= row['m_limit_ratio_08']]
            row['streams'].append(dict(graph=s['graph'], maintenance_cap=s['maintenance_cap'],
                eligible_weight_fraction=sum(2*r['k']*r['n']*r['count'] for r in eligible)/weight,
                eligible_flop_fraction=sum(2*r['m']*r['k']*r['n']*r['count'] for r in eligible)/flops))
        reference.append(row)
    # Unpriced required bandwidth from the preceding reduced model; translate
    # into necessary port counts only, NOT a feasible target architecture.
    required = []
    for s in old['streams']:
        point = next(x for x in s['scenarios'] if x['ratio'] == .8)['requirements'][1]
        rate = point['required_local_bytes_s']
        required.append(dict(graph=s['graph'], maintenance_cap=s['maintenance_cap'],
            old_required_GB_s=rate/1e9,
            old_assumed_near_TFLOPs=point['near_flops_s']/1e12,
            ports=[dict(hb_GB_s=r['hb_GB_s'], minimum_count=math.ceil(rate/(r['hb_GB_s']*1e9)),
                        associated_peak_TFLOPs=math.ceil(rate/(r['hb_GB_s']*1e9))*r['peak_GFLOPs']/1000,
                        controller_only_mm2_40nm=math.ceil(rate/(r['hb_GB_s']*1e9))*r['controller_area_mm2'])
                   for r in reference]))
    samples = []
    for model in source['models']:
        rows = model['rows']
        blocks = [b for row in rows for bank in row['banks'] for b in bank['blocks']]
        stored = sum(b[0] for b in blocks)
        indexed = stored + sum(r['address_table_bytes'] for r in rows)
        samples.append(dict(model=model['model'], blocks=len(blocks), weights=len(blocks)*256,
                            stored_bytes=stored, raw_bytes=len(blocks)*512,
                            indexed_to_raw_ratio=indexed/(len(blocks)*512),
                            min_block_bytes=min(b[0] for b in blocks), max_block_bytes=max(b[0] for b in blocks),
                            bytes_histogram={str(v):sum(b[0]==v for b in blocks) for v in sorted({b[0] for b in blocks})},
                            matrices_with_variable_padded_blocks=sum(len({b[0] for bank in row['banks'] for b in bank['blocks']})>1 for row in rows)))
    result = dict(scope=__doc__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in inputs.items()},
                  paper=dict(pdf_from_jhn='asy/asymmetric_llm_3d/cooling/deep_review/papers/architecture/h2_llm_isca2025.pdf',
                             pdf_sha256='1958e22da94e12eeb225ca5c4694d341416e1567135e2ef53c92f0382217e7f4',
                             pdf_page=10, section='7.1, Table 6', frequency_GHz=1, technology_nm=40,
                             note='GB/s decimal. 16 MAC/FPU; all area numbers remain in paper technology.'),
                  reference=reference, old_requirements=required, samples=samples)
    (root/'evidence/architecture/supply_resource_bounds.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(samples, indent=2))


if __name__ == '__main__':
    main()
