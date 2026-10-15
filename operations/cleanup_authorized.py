"""Remote-only, explicit EDA/chiplet cleanup requested on 2026-10-15."""
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path('/data/ziheng')
assert Path('/home/ziheng').exists() and os.getuid() == 1018
names = '''CATCH CHIPSIM CULS_hh CertifiedChipletPlacement DREAMPlace_fresh
HeteroDSE HeteroDSE-results HeteroDSE-runs MFIT-tree-4444336 Open3DBench
TAP-2.5D-upstream-3a3f4e5 ThermoDSE ThermoDSE-CertiTherm ThermoDSE-CertiTherm_g2_fix
bandsmoke build-env builds certicheck certitherm-gA-20260802-173154
chiplet-inference-artifacts chiplet_baseline chiplet_inference chiplet_repro
composed cor_obs cs_arch_a_resnet50 cs_arch_a_transformer cs_arch_b_resnet50
cs_arch_c_resnet50 csearch csearch_c csverify eda eda_hn edyp edyp6 envcov
experiments fair_compare fcdiag fcverify femcell frontier frontier_tf garnet-gate
geomfac graph_yosys haoning_tr hbt_run heterodse-results heterodse-timeloop
heterodse-timeloop-deps lowsens openroad_build optband paradr_drcu_mp paradr_fork
paradr_haoning paradr_work pdn_gpu placement_profile placeoracle placeoracle2 rbn
rccp remap results rider routedcert routing runs spansweep_b spice_runs splitchk
suiterepro tr_build tts tts_deps wzh wt-1a-schema wt-1b-join wt-stage1a
artifacts git external_repos external_src .pipcache pip_cache pipcache
conda_envs/chiplet-fem-0.11 conda_envs/chiplet-rte
jhn/Placement jhn/proposal jhn/chiplet jhn/detailedrouter
jhn/detailedrouter_out_tracked_backup jhn/spice_batch_fp16
jhn/thermodse-merge jhn/thermodse-upstream-taskpolicy
tools/HotSpot-v7.0 cache/hotspot-superlu-deps cache/fenicsx
vendor/chiplet-link-pathfinding vendor/gemini-hpca2024 vendor/CLIPGen
'''.split()
paths = [BASE / n for n in names]
for p in BASE.iterdir():
    if p.name.startswith(('ChipletThermalEnvelope', 'ecc-', 'bsf-', 'bsf_', 'pdn_gpu')):
        paths.append(p)
for p in (BASE / 'codex-work').iterdir():
    if (p.name.lower().startswith(('moe-chiplet', 'moechiplet', 'thermodse', 'pdc-', 'cts_', 'streaming-'))
            or p.name in ['placement', 'external_baselines', 'tools', 'git-bundles', 'chiplet_research_audit']):
        paths.append(p)
for n in ['jhn_cts', 'cts_build', 'ppopp_ls', 'spice', 'placement_profile', 'HeteroDSE',
          'miniconda3/envs/drcu-build', 'miniconda3/envs/heterodse', '.klayout',
          '.openroad_history', '.yosys_history']:
    paths.append(Path('/home/ziheng') / n)
paths = sorted(set(p for p in paths if p.exists() or p.is_symlink()))
paths = [p for p in paths if not any(q in p.parents for q in paths)]
for p in paths:
    assert p != BASE and p != Path('/home/ziheng')
    assert BASE in p.parents or Path('/home/ziheng') in p.parents
    assert not any(part in ['models', 'hf_cache', 'cleanup-records', 'agent-memory-architecture'] for part in p.parts)
    assert p.parent.resolve() == p.parent, ('symlink ancestor', p)

mounts = [Path(line.split()[4].replace('\\040', ' ')) for line in open('/proc/self/mountinfo')]
for p in paths:
    assert not any(m == p or p in m.parents for m in mounts), ('mounted path', p)

def target_of(s):
    q = Path(s.removesuffix(' (deleted)')) if hasattr(str, 'removesuffix') else Path(s.replace(' (deleted)', ''))
    return next((str(p) for p in paths if q == p or p in q.parents), None)

active = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        if proc.stat().st_uid != os.getuid():
            continue
        links = [proc / 'cwd', proc / 'exe']
        try:
            links += list((proc / 'fd').iterdir())
        except PermissionError:
            pass
        for link in links:
            try:
                value = os.readlink(link)
                target = target_of(value)
                if target:
                    active.append({'pid': proc.name, 'reference': value, 'target': target})
            except (OSError, ValueError):
                pass
    except (OSError, ProcessLookupError):
        pass
assert not active, active

stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
receipt = BASE / 'cleanup-records' / (stamp + '-eda-chiplet')
receipt.mkdir(parents=True)
manifest = {'authorization': 'User: EDA and chiplet related things all delete; reusable caches may delete.',
            'created_utc': stamp, 'active_references': active, 'paths': [],
            'free_before': {p: shutil.disk_usage(p).free for p in ['/data', '/home']}}
for p in paths:
    row = {'path': str(p), 'symlink': p.is_symlink(), 'status': 'pending'}
    if (p / '.git').exists():
        for key, args in [('commit', ['rev-parse', 'HEAD']), ('tracked_status', ['status', '--short', '--untracked-files=no'])]:
            result = subprocess.run(['git', '-C', str(p)] + args, text=True, capture_output=True, timeout=30)
            row[key] = result.stdout[:12000]
    manifest['paths'].append(row)
manifest_path = receipt / 'manifest.json'
def save():
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
save()
print(json.dumps({'receipt': str(manifest_path), 'targets': len(paths)}), flush=True)
assert '--execute' in sys.argv, 'Audit complete; deletion not requested by script flag'
for row in manifest['paths']:
    p = Path(row['path'])
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
        row['status'] = 'removed'
    except OSError as exc:
        row['status'] = 'error'
        row['error'] = str(exc)
    save()
    print(json.dumps({'path': str(p), 'status': row['status']}), flush=True)
manifest['free_after'] = {p: shutil.disk_usage(p).free for p in ['/data', '/home']}
manifest['remaining'] = [row['path'] for row in manifest['paths'] if os.path.lexists(row['path'])]
save()
print(json.dumps({'receipt': str(manifest_path), 'free_before': manifest['free_before'],
                  'free_after': manifest['free_after'], 'remaining': manifest['remaining']}), flush=True)
