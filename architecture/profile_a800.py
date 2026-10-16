"""Remote-only A800 GEMM measurements for trace-derived operator shapes.

Synthetic operands, real M/K/N shapes. Graph and eager timings calibrate the
existing GPU implementation only; no NPU speed/area coefficient is fitted here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    assert str(root).startswith('/data/ziheng/'), 'Run on moe-server, not the local workstation'
    assert not args.output.exists(), 'Use a new evidence destination'
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not active, 'Existing GPU workload: defer this isolated calibration'
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    torch.cuda.set_device(0)
    source=root/'evidence/phase2/workload_accounting.json'
    trace=json.loads(source.read_text())
    cases=[]
    for group in ('mem0_extract_prefix','mem0_update_prefix','embed_full_01_query'):
        calls=sorted(trace['groups'][group]['calls'],key=lambda c:c['prompt_tokens'])
        c=calls[len(calls)//2]
        cfg=trace['models'][c['model']]['config']
        h,ff=cfg['hidden_size'],cfg['intermediate_size']
        q=cfg['num_attention_heads']*cfg['head_dim']
        for op,k,n in [('Q',h,q),('up',h,ff),('down',ff,h)]:
            for phase,m in [('prefill',c['prompt_tokens']),('decode',1)]:
                cases.append(dict(group=group,trace_id=c['trace_id'],model=c['model'],op=op,
                                  phase=phase,m=m,k=k,n=n))
    device=torch.cuda.get_device_properties(0)
    result=dict(scope=__doc__,start=time.time(),status='running',pid=os.getpid(),
                torch=torch.__version__,cuda=torch.version.cuda,device=device.name,
                device_total_bytes=device.total_memory,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(result,indent=2)+'\n'); temp.replace(args.output)
    save()
    try:
        for case in cases:
            m,k,n=case['m'],case['k'],case['n']
            a=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
            b=torch.randn((k,n),device='cuda',dtype=torch.bfloat16)
            c=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
            with torch.inference_mode():
                for _ in range(5): torch.mm(a,b,out=c)
                torch.cuda.synchronize()
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): torch.mm(a,b,out=c)
                torch.cuda.current_stream().wait_stream(stream)
                timings={}
                for mode in ('eager','graph'):
                    samples=[]
                    for _ in range(5):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for repeat in range(10):
                            if mode=='graph': graph.replay()
                            else: torch.mm(a,b,out=c)
                        end.record();end.synchronize()
                        samples.append(begin.elapsed_time(end)/10)
                    timings[mode]=dict(median_ms=statistics.median(samples),samples_ms=samples)
                assert torch.isfinite(c).all().item()
            item=dict(**case,timings=timings,logical_flops=2*m*k*n,
                      compulsory_bytes=2*(m*k+k*n+m*n),
                      scope='Operand/answer floor bytes, not measured HBM traffic; output finiteness only, no new kernel implementation.')
            item['graph_tflops']=item['logical_flops']/(timings['graph']['median_ms']*1e9)
            result['cases'].append(item);save()
            print(case['group'],case['phase'],case['op'],timings['graph']['median_ms'],flush=True)
            del graph,a,b,c
            torch.cuda.empty_cache()
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['end']=time.time();save()


if __name__=='__main__': main()
