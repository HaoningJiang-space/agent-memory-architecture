"""Reconcile the new mechanism results with frozen evidence and resource budgets."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/architecture'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    joint = json.loads((OUT / 'joint_diagnostics.json').read_text())
    checks = []
    for path, expected in joint['input_sha256'].items():
        assert sha(ROOT / path) == expected, path
    checks.append('All joint-result input hashes match current sources and frozen inputs.')
    for path in ('architecture/model.py', 'architecture/spec.json', 'architecture/explore.py',
                 'evidence/architecture/finite_prefix.json'):
        frozen = subprocess.check_output(['git', 'show', '7cd9029:' + path], cwd=ROOT)
        assert frozen == (ROOT / path).read_bytes(), path
    checks.append('Baseline model, spec, lifecycle ledger and paired evidence match 7cd9029 byte for byte.')

    old_update = json.loads((OUT / 'array_diagnosis.json').read_text())
    old_update = next(r['mean'] for r in old_update['results']
                      if r['group'] == 'mem0_update_prefix' and r['case'] == 'baseline')
    old_queries = json.loads((OUT / 'finite_query_model.json').read_text())
    originals = {'update': old_update}
    originals.update({r['strategy'] + '_' + r['strategy']: r['mean']
                      for r in old_queries['profiles']
                      if r['hardware'] == 'HBM3E4_a16_128x128_s128'})
    for row in joint['crossed_controls']:
        if row['mode'] != 'scheduled' or row['bandwidth_factor'] != 1:
            continue
        for key, value in originals[row['group']].items():
            assert math.isclose(row['mean'][key], value, rel_tol=1e-12, abs_tol=1e-8), (row['group'], key)
    checks.append('Original-mode update and paired-query numeric totals reproduce the prior artifacts.')

    spec = json.loads((ROOT / 'architecture/spec.json').read_text())
    assert len(joint['crossed_controls']) == 54
    assert len(joint['designs']) == 96 and len(joint['design_profiles']) == 480
    assert len(joint['mapping_operator_samples']) == 24
    for design in joint['designs']:
        hw = design['hardware']
        for field, budget in [('area_mm2', 'area_mm2'), ('cost_units', 'cost_units'),
                              ('peak_power_w', 'power_w'), ('shoreline_mm', 'shoreline_mm')]:
            assert hw[field] <= spec['budget'][budget], (design['name'], field)
        memory = spec['memory_technology'][hw['memory']]
        assert math.isclose(hw['bandwidth'], memory['bandwidth_GBs'] * 1e9 * hw['packages'], rel_tol=1e-12)
    checks.append('All 96 ranked designs satisfy the declared budgets without diagnostic bandwidth multipliers.')
    for point in joint['attribution']:
        assert math.isclose(point['input_credit_seconds'] + point['output_credit_seconds'],
                            point['points']['raw_raw'] - point['points']['mem0_mem0'], abs_tol=1e-9)
    checks.append('Input and output attribution credits sum to the observed-shape endpoint difference.')

    control = json.loads((OUT / 'representation_controls.json').read_text())
    manifest = control['manifest']
    assert manifest['runner_sha256'] == sha(ROOT / 'architecture/representation_controls.py')
    assert manifest['status'] == control['summary']['status'] == 'complete'
    assert manifest['original_state_unchanged']
    rows = control['summary']['results']
    paired = json.loads((OUT / 'finite_prefix.json').read_text())['summary']['results']
    assert {r['index'] for r in rows} == {r['index'] for r in paired}
    assert len(rows) == 7
    for row in rows:
        assert not row['capped']
        # The final generated token has not been forwarded through the cache.
        assert row['final_live']['kv_bytes'] == (row['input_tokens'] + row['output_tokens'] - 1) * 262144
    checks.append('All seven uncapped GPU controls preserve question coverage, source state and KV accounting.')
    paths = ['architecture/mechanism.py', 'architecture/joint_diagnostics.py',
             'architecture/report_mechanism.py', 'architecture/representation_controls.py',
             'architecture/validate_mechanism.py', 'mechanism_unittest.py',
             'evidence/architecture/joint_diagnostics.json', 'evidence/architecture/representation_controls.json',
             'evidence/architecture/two_case_review.json']
    result = dict(status='passed', checks=checks, crossed_controls=54,
                  ranked_designs=96, design_profiles=480, gpu_queries=7,
                  input_sha256={p: sha(ROOT / p) for p in paths},
                  scope='Artifact consistency and declared-budget checks; not physical-model calibration or quality equivalence.')
    (OUT / 'mechanism_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
