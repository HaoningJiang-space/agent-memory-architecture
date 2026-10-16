"""Independent provenance, sampling-separation and byte-ledger audit."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'evidence/architecture'
    blocks = json.loads((out/'supply_blocks.json').read_text())
    old = json.loads((out/'weight_codec_screen.json').read_text())
    replay = json.loads(gzip.decompress((out/'bank_supply.json.gz').read_bytes()))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert blocks['source_sha256'] == sha(root/'architecture/collect_supply_blocks.py')
    assert blocks['codec_sha256'] == sha(root/'architecture/weight_codec_screen.py')
    assert blocks['reference_sha256'] == sha(out/'weight_codec_screen.json')
    assert replay['source_sha256'] == sha(root/'architecture/bank_supply.py')
    assert replay['input_sha256'] == sha(out/'supply_blocks.json')
    lookup = {}
    count = 0
    for model in blocks['models']:
        previous = next(m for m in old['models'] if m['model'] == model['model'])
        assert previous['revision'] == model['revision']
        for row in model['rows']:
            prior = next(r for r in previous['rows'] if r['tensor'] == row['tensor'])
            assert row['shape'] == prior['shape'] and row['window_lower'] == prior['window_lower']
            n,k = row['shape']
            original_windows = np.linspace(0,n*k-4096,12,dtype=np.int64)
            assert len(row['banks']) == 16
            for index,bank in enumerate(row['banks']):
                lo,hi = (n*index//16)*k,(n*(index+1)//16)*k
                first,last = bank['first_word'],bank['first_word']+32*256
                assert bank['bank'] == index and bank['n_shard_word_bounds'] == [lo,hi]
                assert lo <= first < last <= hi and first % 256 == 0
                assert all(last <= start or first >= start+4096 for start in original_windows)
                assert len(bank['blocks']) == 32
                for stored,valid,escapes,mode in bank['blocks']:
                    assert mode in (0,1) and 0 <= escapes <= 256
                    assert valid == (360+escapes if mode else 520)
                    assert stored == ((valid+63)//64)*64
                    count += 1
            lookup[(model['model'],row['tensor'])] = row
    assert count == 22528 and len(lookup) == 44
    seen = set()
    ledgers = 0
    for entry in replay['results']:
        key = (entry['model'],entry['tensor'],entry['m'],entry['bank_gb_s'])
        assert key not in seen
        seen.add(key)
        row = lookup[key[:2]]
        payload = sum(b[0] for bank in row['banks'] for b in bank['blocks'])
        for label,r in entry['variants'].items():
            assert r['blocks'] == 512 and r['decoded_weight_bytes'] == 512*512
            assert r['bank_bytes'] == payload+row['address_table_bytes']
            assert r['weight_consumer_macs'] == 512*256*entry['m']
            assert r['max_input_credits'] <= 2 and r['max_output_credits'] <= 2
            assert r['input_fifo_provision_bytes'] == 16*2*584
            assert r['output_fifo_provision_bytes'] == 16*2*520
            assert r['crossbar_input_bytes'] == (0 if label == 'local' else payload+512*8)
            assert r['crossbar_output_bytes'] == (0 if label == 'local' else 512*520)
            ledgers += 1
    assert len(seen) == 704 and ledgers == 5632
    receipt = dict(status='pass', fresh_blocks=count, tensors=len(lookup), conditions=len(seen),
                   pipeline_ledgers=ledgers, input_sha256={p.name:sha(p) for p in
                   (out/'supply_blocks.json',out/'bank_supply.json.gz',root/'architecture/validate_supply_evidence.py')},
                   scope='No raw weights on this host; reconstruction checked on server. No PPA or hardware validation.')
    (out/'bank_supply_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
