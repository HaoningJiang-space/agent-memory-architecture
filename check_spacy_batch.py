"""Compare actual GraphRAG texts under serial, pipe and thread scheduling."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import concurrent.futures
import json
from pathlib import Path
import time
import spacy

r=Path(__file__).resolve().parent
api=[json.loads(s) for s in (r/'outputs/api_trace.jsonl').read_text().splitlines()]
source=next(x for x in api if x['context'].get('run')=='graph_full_02' and x['path']=='embeddings')
texts=source['request']['input'][:16]
nlp=spacy.load(json.loads((r/'spacy_model.json').read_text())['path'])
def entities(doc):return [(x.start_char,x.end_char,x.label_,x.text) for x in doc.ents]
report=dict(actual_texts=len(texts),characters=[len(x) for x in texts],scope='16 real GraphRAG fragments; CPU contention present, use timings diagnostically')
begin=time.perf_counter();serial=[entities(nlp(x)) for x in texts];report['serial_seconds']=time.perf_counter()-begin
print('serial',report['serial_seconds'],flush=True)
begin=time.perf_counter();batch=[entities(d) for d in nlp.pipe(texts,batch_size=16,n_process=1)];report['batch_seconds']=time.perf_counter()-begin
report['exact_entities_equal']=serial==batch
print('batch',report['batch_seconds'],report['exact_entities_equal'],flush=True)
(r/'outputs/spacy_batch_check.json').write_text(json.dumps(report,indent=2)+'\n')
