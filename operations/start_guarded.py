"""Wait for phase3 teardown, then try native-prompt ID guards on fresh state.

No concurrent GPU job is introduced. A complete valid 8B run ends this suite;
32B is a bounded structural-validity fallback. Quality is reported separately.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
out = root/'outputs/phase4'
out.mkdir(parents=True,exist_ok=True)
path = out/'pipeline.json'
assert not path.exists()
record = dict(start=time.time(),status='waiting_for_phase3',jobs=[],scope=__doc__)
(out/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())


def save():
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record,indent=2)+'\n')
    temp.replace(path)


def load(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return {}


save()
started = False
try:
    deadline = time.monotonic()+8*3600
    while True:
        prior = load(root/'outputs/phase3/pipeline_recovery.json')
        if prior.get('end'):
            assert prior.get('shutdown',{}).get('remaining')==[], 'Prior teardown is incomplete'
            break
        if time.monotonic()>deadline: raise TimeoutError('Phase3 completion deadline exceeded')
        time.sleep(10)
    # Do not silently compete with a new unrelated GPU job after the wait.
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not active, 'Other GPU processes are active; no new services started'
    for port in (8030,8031,8032,8033):
        with socket.socket() as test:
            test.bind(('127.0.0.1',port))
    subprocess.run([sys.executable,'launch_services.py','--session','phase4','--prefix-cache',
                    'proxy','embedding','construction','reasoning'],cwd=root,check=True)
    started = True
    deadline = time.monotonic()+900
    for port in (8030,8031,8032,8033):
        while True:
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3).read()
                break
            except OSError:
                if time.monotonic()>deadline: raise TimeoutError('Guarded service readiness timeout')
                time.sleep(5)
    for size in ('8B','32B'):
        name = 'mem0_'+size.lower()+'_guarded'
        command = [sys.executable,'-u','run_native.py','--algorithm','mem0','--run','phase4/'+name,
                   '--stage','both','--construction-model','Qwen/Qwen3-'+size,
                   '--answer-max-tokens','1024','--trace-file','outputs/phase4/api_trace.jsonl',
                   '--update-control','guarded_native']
        job = dict(name=name,command=command,start=time.time(),status='running')
        record['jobs'].append(job)
        record['status']='running'
        save()
        with (out/(name+'.log')).open('xb') as log:
            result = subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=False)
        manifest = load(out/name/'both_manifest.json')
        job.update(end=time.time(),returncode=result.returncode,status=manifest.get('status','missing_manifest'),
                   validity=manifest.get('validity'),error=manifest.get('error'))
        save()
        if manifest.get('status')=='complete' and manifest.get('validity')=='valid':
            sys.path.insert(0,str(root/'external/agent-memory-profiling'))
            from utils.eval_other_utils import post_process
            answers = [json.loads(line) for line in (out/name/'answers.jsonl').read_text().splitlines()]
            assert len(answers)==manifest['total_queries']==60
            metrics = [post_process(r['response'],r['answer'] if isinstance(r['answer'],list)
                       else [r['answer']],manifest['dataset_config'])[0] for r in answers]
            job['lexical_metrics']={k:statistics.mean(r[k] for r in metrics) for k in metrics[0]}
            job['quality_scope']='Lexical metrics; no independent semantic judge or same-quality claim.'
            capacity = subprocess.run([sys.executable,'capacity_tradeoff.py','--run','phase4/'+name,
                                       '--name',name+'_capacity'],cwd=root,check=False)
            job['capacity_returncode']=capacity.returncode
            save()
            break
    record['status']='experiments_collected'
except BaseException as exc:
    record.update(status='failed',error=repr(exc))
    raise
finally:
    if started:
        services_path=root/'services_phase4.json'
        services=load(services_path)
        (out/'service_configuration.json').write_text(json.dumps(services,indent=2)+'\n')
        signalled=[]
        for service in services.values():
            pid=service['pid']; process=Path('/proc')/str(pid)
            if not process.exists(): continue
            assert process.stat().st_uid==os.getuid()
            actual=process.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            assert actual==service['argv'] and os.getpgid(pid)==pid
            os.killpg(pid,signal.SIGTERM);signalled.append(pid)
        until=time.monotonic()+30
        remaining=signalled
        while remaining and time.monotonic()<until:
            time.sleep(1)
            active=[]
            for pid in remaining:
                try:
                    if (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z': active.append(pid)
                except FileNotFoundError: pass
            remaining=active
        record['shutdown']=dict(signalled=signalled,remaining=remaining)
        if not remaining: services_path.write_text('{}\n')
    files={}
    for file in sorted(out.rglob('*')):
        if not file.is_file() or file.is_symlink() or file in (path,out/'raw_sha256.json'): continue
        digest=hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        files[str(file.relative_to(root))]=dict(bytes=file.stat().st_size,sha256=digest.hexdigest())
    (out/'raw_sha256.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    record['end']=time.time()
    save()
