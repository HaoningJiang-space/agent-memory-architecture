"""Mirror-only download of the pinned LightMem encoder; no corpus upload."""
import hashlib
import json
import os
from pathlib import Path
import requests

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
NAME = 'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank'


def main():
    info = HfApi(endpoint=os.environ['HF_ENDPOINT']).model_info(NAME, files_metadata=True)
    path = snapshot_download(NAME, revision=info.sha, endpoint=os.environ['HF_ENDPOINT'],
        cache_dir=str(ROOT / 'hf/hub'), max_workers=2,
        allow_patterns=['config.json', 'tokenizer*', 'special_tokens_map.json',
                        'vocab.txt', 'added_tokens.json'])
    # The HF mirror can redirect large files to an unreachable CAS endpoint.
    # Fetch the public mirror copy, but accept it only if its bytes match the
    # pinned HF revision's LFS digest. Do not silently substitute a new model.
    weight = next(f for f in info.siblings if f.rfilename == 'model.safetensors')
    expected = weight.lfs.sha256
    target = Path(path) / 'model.safetensors'
    if not target.exists():
        url = 'https://modelscope.cn/models/' + NAME + '/resolve/master/model.safetensors'
        temp = ROOT / 'outputs/encoder_model.part'
        digest = hashlib.sha256()
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temp.open('wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError('Mirror weights differ from pinned HF model')
        temp.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError('Existing weights do not match pinned HF model')
    files = {f.name: dict(bytes=f.stat().st_size,
                         sha256=hashlib.sha256(f.read_bytes()).hexdigest())
             for f in Path(path).iterdir() if f.is_file()}
    (ROOT / 'outputs/compressor_source.json').write_text(json.dumps(dict(
        model=NAME, revision=info.sha, path=path, mirror=os.environ['HF_ENDPOINT'],
        weight_mirror='https://modelscope.cn', pinned_weight_sha256=expected,
        files=files), indent=2) + '\n')
    print('complete', info.sha, list(files))


if __name__ == '__main__':
    main()
