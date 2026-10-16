"""Start one owned, bounded calibration process in the background."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True)
    args = p.parse_args()
    assert args.session.replace('_', '').isalnum()
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    record = logs / f'{args.session}_process.json'
    if record.exists() or (ROOT / 'outputs' / args.session).exists():
        raise FileExistsError(args.session)
    argv = [sys.executable, '-m', 'architecture.capacity_batch_probe', '--session', args.session]
    with (logs / f'{args.session}.log').open('x') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(pid=proc.pid, argv=argv, started=time.time(), session=args.session)
    record.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
