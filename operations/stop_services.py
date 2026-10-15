"""Release only the exact isolated service groups launched for this pilot."""
import json
import os
from pathlib import Path
import signal
import time

root = Path(__file__).resolve().parents[1]
assert json.loads((root/'outputs/completion_pipeline.json').read_text())['status']=='complete'
assert json.loads((root/'outputs/validation.json').read_text())['checks_passed']
manifest_path = root/'services.json'
services = json.loads(manifest_path.read_text())
(root/'outputs/service_configuration.json').write_text(json.dumps(services,indent=2)+'\n')
report = dict(start=time.time(),services=[])
for name, service in services.items():
    pid = service['pid']
    process = Path('/proc')/str(pid)
    if not process.exists():
        report['services'].append(dict(name=name,pid=pid,status='already_exited'))
        continue
    assert process.stat().st_uid==os.getuid(), 'Service owner changed'
    actual = process.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    assert actual==service['argv'], 'PID no longer matches recorded service'
    assert os.getpgid(pid)==pid, 'Not an isolated service group'
    os.killpg(pid,signal.SIGTERM)
    report['services'].append(dict(name=name,pid=pid,status='SIGTERM_sent'))
deadline = time.monotonic()+20
while time.monotonic()<deadline:
    active = []
    for service in services.values():
        status = Path('/proc')/str(service['pid'])/'stat'
        try:
            state = status.read_text().rsplit(')',1)[1].split()[0]
            if state!='Z':
                active.append(service['pid'])
        except FileNotFoundError:
            pass
    if not active:
        break
    time.sleep(1)
report.update(end=time.time(),remaining_parent_pids=active)
(root/'outputs/service_shutdown.json').write_text(json.dumps(report,indent=2)+'\n')
assert not active,'Some services remain; inspect without broad process termination'
manifest_path.write_text('{}\n')
print(json.dumps(report,indent=2))
