"""Run three bounded, sequential path controls; never overlap GPU cases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='lightpath')
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    if not args.prefix.replace('_', '').isalnum():
        raise ValueError('Unsafe session prefix')
    cases = [('flat', 512), ('full', 512), ('full', 768)]
    status = ROOT / 'outputs' / (args.prefix + '_cases_status.json')
    if status.exists():
        raise FileExistsError(status)
    rows = []
    def save():
        status.write_text(json.dumps(rows, indent=2) + '\n')
    for variant, stm in cases:
        session = f'{args.prefix}_{variant}{stm}'
        row = dict(session=session, variant=variant, stm_tokens=stm,
                   start=time.time(), status='starting')
        rows.append(row)
        save()
        try:
            for port in [8031, 8032, 8033]:
                with urlopen(f'http://127.0.0.1:{port}/health', timeout=20) as response:
                    assert response.status == 200
            # Eliminate cross-case prefix-cache reuse. These are isolated services.
            for port in [8031, 8032]:
                request = Request(f'http://127.0.0.1:{port}/reset_prefix_cache', method='POST')
                with urlopen(request, timeout=20) as response:
                    assert response.status == 200
            argv = [sys.executable, '-m', 'architecture.lightmem_path_probe',
                    '--variant', variant, '--stm-tokens', str(stm), '--session', session]
            row.update(status='running', argv=argv)
            save()
            with (ROOT / 'logs' / (session + '.log')).open('x') as log:
                result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout)
            row['returncode'] = result.returncode
            if result.returncode:
                raise RuntimeError(f'{session} failed; inspect its log and manifest')
            manifest = json.loads((ROOT / 'outputs' / session / 'lightmem/manifest.json').read_text())
            assert manifest['status'] == 'complete'
            row.update(status='complete', end=time.time())
            save()
        except Exception as exc:
            row.update(status='failed', end=time.time(), error=f'{type(exc).__name__}: {exc}')
            save()
            raise


if __name__ == '__main__':
    main()
