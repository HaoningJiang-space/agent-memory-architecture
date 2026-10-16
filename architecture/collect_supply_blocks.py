"""Collect fresh contiguous block metadata from existing remote BF16 weights.

No weight values/bitstreams are exported. Each N shard contributes a contiguous
run; all original codec train/test windows are excluded. This is a bounded,
deterministic sample, not a whole-model or physical DRAM address trace.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

import weight_codec_screen
from weight_codec_screen import BLOCK, address_table_bytes, decode, encode, headers


def select_start(lo, hi, count, excluded):
    start = math.ceil(lo / BLOCK) * BLOCK
    for left, right in sorted(excluded):
        if start + count <= left:
            break
        if start < right and start + count > left:
            start = math.ceil(right / BLOCK) * BLOCK
    if start + count > hi:
        raise ValueError('No fresh contiguous run inside N shard')
    return start


def collect(directory, previous, banks=16, blocks_per_bank=32):
    tensors, hashes = headers(directory)
    if hashes != previous['header_sha256']:
        raise ValueError('Checkpoint headers differ from frozen codec screen')
    rows = []
    for reference in previous['rows']:
        path, offset, header = tensors[reference['tensor']]
        if header['dtype'] != 'BF16' or header['shape'] != reference['shape']:
            raise ValueError('Tensor changed')
        n, k = header['shape']  # Torch stores W as [output, input].
        count = n * k
        excluded = [(int(i), int(i) + 4096) for i in
                    np.linspace(0, count - 4096, 12, dtype=np.int64)]
        data = np.memmap(path, dtype='<u2', mode='r',
                         offset=offset + header['data_offsets'][0], shape=(count,))
        bank_rows = []
        for bank in range(banks):
            lo, hi = (n * bank // banks) * k, (n * (bank + 1) // banks) * k
            start = select_start(lo, hi, BLOCK * blocks_per_bank, excluded)
            words = np.array(data[start:start + BLOCK * blocks_per_bank])
            records = []
            for i in range(blocks_per_bank):
                original = words[i * BLOCK:(i + 1) * BLOCK]
                blob = encode(original, reference['window_lower'])
                if not np.array_equal(original, decode(blob)):
                    raise ValueError('Bit-exact reconstruction failed')
                _, mode, escapes, size = struct.unpack('<BBHI', blob[:8])
                valid = 8 + (size * 2 if mode == 0 else 3 * size // 8 + size + escapes)
                records.append([len(blob), valid, escapes, mode])
            bank_rows.append(dict(bank=bank, n_shard_word_bounds=[lo, hi],
                                  first_word=start, sample_sha256=hashlib.sha256(words.tobytes()).hexdigest(),
                                  blocks=records))
        del data
        rows.append(dict(tensor=reference['tensor'], shape=[n, k],
                         window_lower=reference['window_lower'], banks=bank_rows,
                         address_table_bytes=address_table_bytes(banks * blocks_per_bank),
                         bit_exact=True))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.models.read_text())
    reference = json.loads(args.reference.read_text())
    result = dict(scope=__doc__, banks=16, blocks_per_bank=32, weights_per_block=BLOCK,
                  fields=['stored_bytes', 'valid_bytes', 'escape_weights', 'mode'],
                  models=[], numpy_version=np.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  codec_sha256=hashlib.sha256(Path(weight_codec_screen.__file__).read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    for old in reference['models']:
        model = old['model']
        if manifest[model]['revision'] != old['revision']:
            raise ValueError('Checkpoint revision changed')
        rows = collect(Path(manifest[model]['path']), old)
        result['models'].append(dict(model=model, revision=old['revision'], rows=rows))
        print(model, len(rows), 'matrices checked', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
