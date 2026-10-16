"""Sequential background software controls, complete histories, and exact-PID teardown."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--recover',action='store_true',help='Fresh complete runs after the audited receipt-race fix')
args = parser.parse_args()
out = root/'outputs/phase3'
out.mkdir(parents=True,exist_ok=True)
path = out/('pipeline_recovery.json' if args.recover else 'pipeline.json')
assert not path.exists(), 'Use a fresh session; never overwrite run evidence'
record = dict(start=time.time(),status='waiting',jobs=[],
    scope='Sequential software controls on one real history. No cross-job GPU contention. '
          'Shared host CPU and differing cache state preclude isolated hardware rankings. '
          'Flat protocol is an explicit software variant; quality must be evaluated separately.')


def save():
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record,indent=2)+'\n')
    temp.replace(path)


def job(name, command):
    entry = dict(name=name,command=command,start=time.time(),status='running')
    record['jobs'].append(entry)
    record['status'] = 'running'
    save()
    with (out/f'{name}.log').open('xb') as log:
        result = subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=False)
    entry.update(end=time.time(),returncode=result.returncode,
                 status='complete' if result.returncode == 0 else 'failed')
    save()
    return entry


save()
try:
    deadline = time.monotonic()+900
    for port in (8030,8031,8032,8033):
        while True:
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3).read()
                break
            except OSError:
                if time.monotonic()>deadline:
                    raise RuntimeError(f'Port {port} readiness timeout')
                time.sleep(5)
    record['ready'] = time.time()
    if not args.recover:
        job('grammar',[sys.executable,'-u','grammar_replay.py'])
        job('frozen_protocol',[sys.executable,'-u','protocol_ablation.py','--session','phase3',
                               '--variants','native','flat_snapshot','flat_delta'])
    for name, model, control in [('mem0_8b_delta','Qwen/Qwen3-8B','flat_delta'),
                                  ('mem0_8b_snapshot','Qwen/Qwen3-8B','flat_snapshot'),
                                  ('mem0_32b_delta','Qwen/Qwen3-32B','flat_delta')]:
        if args.recover:
            name += '_v2'
        entry = job(name,[sys.executable,'-u','run_native.py','--algorithm','mem0',
                          '--run',f'phase3/{name}','--stage','both',
                          '--construction-model',model,'--answer-max-tokens','1024',
                          '--trace-file','outputs/phase3/api_trace.jsonl','--update-control',control])
        manifest_path = out/name/'both_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get('status') == 'complete' and manifest.get('validity') == 'valid':
            sys.path.insert(0,str(root/'external/agent-memory-profiling'))
            from utils.eval_other_utils import post_process
            answers = [json.loads(line) for line in (out/name/'answers.jsonl').read_text().splitlines()]
            assert len(answers) == manifest['total_queries'] == 60
            metrics = [post_process(r['response'],r['answer'] if isinstance(r['answer'],list)
                       else [r['answer']],manifest['dataset_config'])[0] for r in answers]
            entry['lexical_metrics'] = {k:statistics.mean(r[k] for r in metrics) for k in metrics[0]}
            entry['quality_scope'] = 'Lexical metrics only; no independent semantic judge.'
        else:
            entry['failure'] = manifest.get('error','Missing manifest')
        if manifest.get('status') == 'complete' and manifest.get('validity') == 'valid':
            job(name+'_capacity',[sys.executable,'-u','capacity_tradeoff.py',
                                  '--run',f'phase3/{name}','--name',name+'_capacity'])
        save()
    record['status'] = 'experiments_collected'
except BaseException as exc:
    record.update(status='failed',error=repr(exc))
    raise
finally:
    services_path = root/'services_phase3.json'
    services = json.loads(services_path.read_text())
    (out/'service_configuration.json').write_text(json.dumps(services,indent=2)+'\n')
    terminated = []
    for name, service in services.items():
        pid = service['pid']
        process = Path('/proc')/str(pid)
        if not process.exists():
            continue
        assert process.stat().st_uid == os.getuid()
        actual = process.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        assert actual == service['argv'] and os.getpgid(pid) == pid, 'PID identity changed; refuse signal'
        os.killpg(pid,signal.SIGTERM)
        terminated.append(pid)
    until = time.monotonic()+30
    remaining = terminated
    while remaining and time.monotonic()<until:
        time.sleep(1)
        active = []
        for pid in remaining:
            try:
                if (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0] != 'Z':
                    active.append(pid)
            except FileNotFoundError:
                pass
        remaining = active
    record['shutdown'] = dict(signalled=terminated,remaining=remaining)
    if not remaining:
        services_path.write_text('{}\n')
    files = {}
    for file in sorted(out.rglob('*')):
        if not file.is_file() or file.is_symlink() or file == path:
            continue
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        files[str(file.relative_to(root))] = dict(bytes=file.stat().st_size,sha256=digest.hexdigest())
    (out/'raw_sha256.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    record['end'] = time.time()
    save()
