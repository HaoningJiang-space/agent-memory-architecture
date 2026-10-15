"""Remove inspected EDA/chiplet temporary files after the main cleanup."""
import datetime
import json
import os
from pathlib import Path
import shutil

base = Path('/data/ziheng')
names = '''tmp jhn/conda_jhn jhn/sat_solvers .cuda_cache .cupy_cache cupy_cache
__pycache__ torch_extensions build_cache worktrees vtmp envs
alpha_sweep_agg.py band_smoke.json build_convert.sh build_tr.sh clone_dreamplace.sh
commit_policy_compare.py composed.log cs_arch_a_resnet50.log cs_arch_a_transformer.log
cs_arch_b_resnet50.log cs_arch_c_resnet50.log csearch.log csearch2.log csearch_c.log
def_sketchability.py diff_eq_layer2.py diff_equilibrium_layer.py disagreement_band.json
dr_conflict_probe.py dreamplace_clone.log edyp2.log edyp6.log edyp_probe.log
edyp_probe.py edyp_probe6.py envcov2.log envelope_coverage.json envelope_coverage2.json
field_preconditioned_assign.py fp_hard.py geomfac.log geomfac_arch_b.log geomfac_arch_c.log
gr_ariane.sh heterodse-timeloop-build.log iter_stability.py killgate_mol.sh ledger_replay.py
limitparam.json limitparam2.json limitparam3.json lowsens.log lowsens2.log
mk_orbuild_env.sh o3d_pathb_probe.tcl o3d_pull.log o3d_pull2.log o3d_setup.log optband.log
optimum_band.json or_cmake.log orbuild_env.log paradr_eval.sh price_ledger.json
pull_o3d.sh remap.log route_mol_bpfe.sh run_lol_one.sh run_mazeprofile.sh run_mol_one.sh
run_mp2.sh run_ruleblind.sh run_specrun.sh setup_o3d.sh spansweep2.log spansweep_b.log
suite_repro.json suiterepro.log sweep_lol_bpfe.sh test_hbt_access.sh
timeloop-oaves-gcc9-boost171.patch toy_priced_placement.py tr_cmake.log
'''.split()
paths = [base / n for n in names]
paths += [p for p in base.iterdir() if p.name.startswith('ruleblind')]
paths = [p for p in paths if os.path.lexists(p)]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        links = [proc / 'cwd', proc / 'exe']
        try:
            links += list((proc / 'fd').iterdir())
        except OSError:
            pass
        for link in links:
            try:
                q = Path(os.readlink(link))
                assert not any(q == p or p in q.parents for p in paths), str(link)
            except OSError:
                pass
    except OSError:
        pass
receipt = base / 'cleanup-records' / '20261015T142650Z-eda-chiplet' / 'residuals.json'
result = {'free_before': shutil.disk_usage('/data').free, 'paths': []}
for p in paths:
    assert base in p.parents and p.parent.resolve() == p.parent
    row = {'path': str(p)}
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
        row['status'] = 'removed'
    except OSError as exc:
        row['status'] = 'error'
        row['error'] = str(exc)
    result['paths'].append(row)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
result['free_after'] = shutil.disk_usage('/data').free
receipt.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'receipt': str(receipt), 'counts': {k:sum(r['status']==k for r in result['paths']) for k in ['removed','error']}, 'errors':[r for r in result['paths'] if r['status']=='error'], 'free_before':result['free_before'], 'free_after':result['free_after']}))
