"""Hash finished remote evidence without exporting raw workload contents."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
output = root/'outputs'
excluded = {'raw_artifact_sha256.json'}
files = [p for p in output.rglob('*') if p.is_file() and not p.is_symlink()
         and 'summaries' not in p.relative_to(output).parts and p.name not in excluded]
files += [root/name for name in ['models.json','dataset.json','longmemeval.json',
          'spacy_model.json','nltk_sources.json','tiktoken_cl100k_source.json','services.json']]
entries = {}
for path in sorted(files):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    entries[str(path.relative_to(root))] = dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
record = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='Finished remote artifact snapshot; future reruns require a new receipt. Raw contents remain remote.',
    files=entries)
(output/'raw_artifact_sha256.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(files=len(entries),bytes=sum(x['bytes'] for x in entries.values()))))
