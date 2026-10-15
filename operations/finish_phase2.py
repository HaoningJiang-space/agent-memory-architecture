"""Background collection and exact-PID teardown for the two registered phase-2 jobs."""
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
out = root/'outputs/phase2'
target = out/'completion.json'
if target.exists():
    raise RuntimeError('Completion receipt already exists')
record = dict(start=time.time(), status='waiting')
target.write_text(json.dumps(record,indent=2)+'\n')


def load_if_ready(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


try:
    deadline = time.monotonic()+8*3600
    while True:
        native = load_if_ready(out/'native_control_pipeline.json')
        protocol = load_if_ready(out/'protocol_ablation/manifest.json')
        if all(x.get('status') in ('complete','failed') for x in (native,protocol)):
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Registered jobs did not finish in 8 hours; services retained for inspection')
        time.sleep(10)
    record.update(status='collecting', native_status=native['status'], protocol_status=protocol['status'])
    target.write_text(json.dumps(record,indent=2)+'\n')
    subprocess.run([sys.executable,str(root/'phase2_snapshot.py'),'--name','completed_observation'],
                   cwd=root,check=True)
    run = out/'mem0_32b_native'
    manifest = json.loads((run/'both_manifest.json').read_text())
    answers = run/'answers.jsonl'
    if manifest['status'] == 'complete' and manifest['validity'] == 'valid':
        sys.path.insert(0,str(root/'external/agent-memory-profiling'))
        from utils.eval_other_utils import post_process
        rows = [json.loads(line) for line in answers.read_text().splitlines()]
        assert len(rows) == manifest['total_queries'] == 60
        metrics = []
        for row in rows:
            truth = row['answer'] if isinstance(row['answer'],list) else [row['answer']]
            metric, _ = post_process(row['response'],truth,manifest['dataset_config'])
            metrics.append(metric)
        record['answer_metrics'] = dict(count=len(rows),
            means={k:statistics.mean(x[k] for x in metrics) for k in metrics[0]},
            scope='Pinned lexical post_process only; no independent semantic judge')
    else:
        record['answer_metrics'] = dict(status='not_valid_for_quality_comparison',
                                        reason=manifest.get('error',manifest['validity']))
    record['scope'] = ('Completion of the registered experiments, not completion of the research motivation. '
                       'Overlapping GPU0 embedding/protocol work prevents uncontended latency/energy claims.')
    services_path = root/'services_phase2.json'
    services = json.loads(services_path.read_text())
    (out/'service_configuration.json').write_text(json.dumps(services,indent=2)+'\n')
    if (out/'keep_services').exists():
        record['services'] = 'retained by keep_services marker'
    else:
        terminated = []
        for name, service in services.items():
            pid = service['pid']
            process = Path('/proc')/str(pid)
            if not process.exists():
                continue
            assert process.stat().st_uid == os.getuid()
            actual = process.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            assert actual == service['argv'], f'PID reused for {name}; refuse signal'
            assert os.getpgid(pid) == pid
            os.killpg(pid,signal.SIGTERM)
            terminated.append(pid)
        until = time.monotonic()+30
        remaining = terminated
        while remaining and time.monotonic() < until:
            time.sleep(1)
            active = []
            for pid in remaining:
                try:
                    state = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
                    if state != 'Z':
                        active.append(pid)
                except FileNotFoundError:
                    pass
            remaining = active
        record['service_shutdown'] = dict(signalled_pids=terminated,remaining_pids=remaining)
        assert not remaining, 'Service shutdown incomplete'
        services_path.write_text('{}\n')
    files = {}
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path.is_symlink() or path in (target,out/'raw_sha256.json'):
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        files[str(path.relative_to(root))] = dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
    (out/'raw_sha256.json').write_text(json.dumps(dict(scope='Phase-2 files after registered jobs finished',files=files),indent=2)+'\n')
    record['status'] = 'complete'
except Exception as exc:
    record.update(status='failed',error=repr(exc))
    raise
finally:
    record['end'] = time.time()
    target.write_text(json.dumps(record,indent=2)+'\n')
