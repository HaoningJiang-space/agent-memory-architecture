"""Stop/restart only exact recorded service groups in this isolated checkout."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stop', 'restart'])
    parser.add_argument('--session', required=True)
    parser.add_argument('--names', nargs='+', default=['proxy', 'embedding', 'construction', 'reasoning'])
    parser.add_argument('--dev-cache-reset', action='store_true')
    args = parser.parse_args()
    if not args.session.replace('_', '').isalnum():
        raise ValueError('Unsafe session')
    path = ROOT / f'services_{args.session}.json'
    services = json.loads(path.read_text())
    stopped = {}
    for name in args.names:
        service = services[name]
        pid = service['pid']
        proc = Path('/proc') / str(pid)
        if proc.exists():
            assert proc.stat().st_uid == os.getuid()
            actual = proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            assert actual == service['argv'], 'PID does not match recorded command'
            assert os.getpgid(pid) == pid, 'Not an isolated process group'
            os.killpg(pid, signal.SIGTERM)
        stopped[name] = service
    deadline = time.monotonic() + 25
    while True:
        remaining = []
        for name, service in stopped.items():
            p = Path('/proc') / str(service['pid']) / 'stat'
            if p.exists() and p.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                remaining.append(name)
        if not remaining:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Services still stopping: {remaining}')
        time.sleep(.5)
    receipt = ROOT / 'outputs' / f'{args.session}_{args.action}_{time.time_ns()}.json'
    receipt.write_text(json.dumps(stopped, indent=2) + '\n')
    for name in stopped:
        del services[name]
    path.write_text(json.dumps(services, indent=2) + '\n')
    if args.action == 'restart':
        env = dict(os.environ)
        if args.dev_cache_reset:
            env['VLLM_SERVER_DEV_MODE'] = '1'
        subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'launch_services.py'),
                        '--session', args.session, '--prefix-cache', *stopped],
                       cwd=ROOT, env=env, check=True)
    print(receipt)


if __name__ == '__main__':
    main()
