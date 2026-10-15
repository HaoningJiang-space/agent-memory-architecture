"""Analyze measured launch controls; retain a falsifiable holdout error gate."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

root = Path(__file__).resolve().parent
raw = json.loads((root/'outputs/embedding_graph_calibration.json').read_text())
native = json.loads((root/'outputs/embedding_calibration.json').read_text())
cfg = native['config']
linear_parameters = sum(np.prod(s['weight']) for s in native['cases'][0]['linear_shapes'])

def features(row):
    b,n = row['batch'],row['length']
    # Logical FLOPs, not hardware counters. Dense causal attention pair count.
    dense = 2*b*n*linear_parameters
    attn = 2*b*n*(n+1)*cfg['num_hidden_layers']*cfg['num_attention_heads']*cfg['head_dim']
    return [1.0,dense/1e12,attn/1e12]

x=np.array([features(r) for r in raw['cases']])
y=np.array([r['graph']['median_seconds'] for r in raw['cases']])
predictions=[]
for i in range(len(y)):
    mask=np.arange(len(y))!=i
    coefficient,_=nnls(x[mask],y[mask])
    predictions.append(float(x[i]@coefficient))
error=np.abs(np.array(predictions)-y)/y
coeff,_=nnls(x,y)
report=dict(model='nonnegative launch + dense FLOP + attention FLOP shape fit',
            purpose='within-A800 calibration only; no new-hardware prediction',
            validation='leave-one-shape-out; 8 shapes, deterministic synthetic inputs',
            gate=dict(median_relative_error_limit=0.20,max_relative_error_limit=0.35,
                      median_relative_error=float(np.median(error)),max_relative_error=float(max(error)),
                      passed=bool(np.median(error)<=0.20 and max(error)<=0.35)),
            coefficients=dict(overhead_s=float(coeff[0]),dense_s_per_TFLOP=float(coeff[1]),attention_s_per_TFLOP=float(coeff[2])),
            logical_linear_weight_bytes=int(linear_parameters*2),
            parameters=native['parameters'], cases=[])
for i,r in enumerate(raw['cases']):
    report['cases'].append(dict(batch=r['batch'],length=r['length'],
        eager_ms=r['eager']['median_seconds']*1000,graph_ms=y[i]*1000,
        speedup=r['speedup'],heldout_prediction_ms=predictions[i]*1000,
        heldout_relative_error=float(error[i]),logical_dense_flops=float(x[i,1]*1e12),
        logical_attention_flops=float(x[i,2]*1e12)))
(root/'outputs/calibration_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
