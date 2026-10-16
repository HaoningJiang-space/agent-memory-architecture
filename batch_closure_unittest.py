import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from architecture import model as base,mechanism as mech,supply_path as supply,small_batch as small
from architecture.batch_closure import graph,replay
from architecture.collect_family import records,collect
from architecture.explore import load
from architecture.family_replay import batch_capacity,batched


class BatchClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec,cls.trace,_,capacity,_=load();cls.weights={m['model']:m['weight_bytes'] for m in capacity['models']}
        hw=next(h for h in base.candidates(spec)[0] if h.name=='HBM3E4_a16_128x128_s128')
        cls.hw,_,_=mech.provision(hw,mech.KShard(groups=2),spec)
        cls.cfg=cls.trace['models']['Qwen/Qwen3-8B']['config']

    def test_missing_trace_fails_instead_of_free_embedding(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):records(Path(d)/'not-recorded.jsonl')

    def test_wrong_session_empty_embedding_is_not_accepted(self):
        with TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();service=root/'service';service.mkdir()
            keys=['algorithm','status','start','end','runner_sha256','dataset_sha256','history_sha256',
                  'source_sha256','native_revision','records','versions','state_disk_bytes','selection',
                  'quality_scope','deployment']
            manifest={k:None for k in keys};manifest.update(algorithm='simplemem',status='complete',start=0,end=2)
            (run/'manifest.json').write_text(json.dumps(manifest))
            for name in ('llm_private.jsonl','operations.jsonl','state_private.json'):(run/name).write_text('')
            for name in ('api_trace.jsonl','embedding_kernels.jsonl'):(service/name).write_text('')
            with self.assertRaisesRegex(ValueError,'No embedding evidence'):collect(run,service)

    def test_small_m_math_and_weight_reuse(self):
        rng=np.random.default_rng(14)
        for m in (1,2,4,8):
            x=rng.normal(size=(m,17));w=rng.normal(size=(17,19))
            np.testing.assert_allclose(small.execute_reference(x,w,7),x@w,atol=2e-6,rtol=2e-6)
            r=small.sharded(self.hw,m,4096,12288)
            self.assertEqual(r['flops'],2*m*4096*12288)
            self.assertEqual(r['weight_read_bytes'],2*4096*12288)
            self.assertEqual(r['partial_buffer_bytes'],2*4*2*m*self.hw.cols)
            self.assertLessEqual(r['shared_tile_bytes'],(self.hw.sram_mib*base.MIB-supply.dma_buffer_bytes(self.hw))/self.hw.arrays)

    def test_legacy_m1_agrees_and_large_m_falls_back(self):
        old=mech.sharded(supply.tile_hardware(self.hw),mech.KShard(groups=2),1,4096,12288)
        old=supply.score(self.hw,dict(old,output_columns=12288,counted_k_device=True))
        new=small.sharded(self.hw,1,4096,12288)
        for field in supply.FIELDS:self.assertAlmostEqual(old[field],new[field],delta=max(abs(old[field])*1e-12,1e-8))
        a=small.linear(self.hw,29,4096,12288,'small_m',supply.LocalPath(),.3)
        b=supply.linear(self.hw,29,4096,12288,True)
        self.assertEqual(a,b)

    def test_capacity_rejection_and_local_fifo_bound(self):
        calls=[dict(id='one',L=17,O=3)]
        with self.assertRaises(ValueError):batched(replace(self.hw,capacity_bytes=1),self.cfg,calls)
        required=batch_capacity(self.cfg,calls)
        with self.assertRaises(ValueError):small.execute_batch(replace(self.hw,capacity_bytes=required-1),self.cfg,calls,'baseline')
        path=supply.LocalPath()
        for m in (1,4,8):
            r=small.sharded(self.hw,m,4096,12288,path,.3)
            self.assertLessEqual(r['hb_weight_tile_bytes'],r['hb_half_fifo_per_array'])

    def test_dependency_ready_and_no_eight_request_cap(self):
        nodes=[dict(id=f'root{i}',kind='host',deps=[],seconds=0,stage='barrier') for i in range(12)]
        nodes += [dict(id=f'c{i}',kind='llm',deps=[f'root{i}'],model='Qwen/Qwen3-8B',L=5,O=3,stage='maintenance') for i in range(12)]
        nodes += [dict(id='child',kind='llm',deps=['c0'],model='Qwen/Qwen3-8B',L=7,O=2,stage='query')]
        dag=dict(nodes=nodes,query_ends=['child'])
        r=replay(dag,self.hw,'baseline',self.trace,self.weights)
        self.assertEqual(r['batches'][0]['batch'],12)
        self.assertGreaterEqual(r['node_starts']['child'],r['node_completions']['c0'])
        faster=replay(dag,replace(self.hw,frequency=self.hw.frequency*2),'baseline',self.trace,self.weights)
        self.assertNotEqual(r['node_starts']['child'],faster['node_starts']['child'])

    def test_exact_terminal_event_completes_without_false_deadlock(self):
        dag=dict(nodes=[dict(id='host',kind='host',deps=[],seconds=1.,stage='terminal')],query_ends=['host'])
        r=replay(dag,self.hw,'baseline',self.trace,self.weights)
        self.assertEqual(r['seconds'],r['initial_load_seconds']+1.)

    def test_recorded_graph_consumes_every_call_and_has_no_api_arrivals(self):
        data=json.loads(Path('evidence/architecture/family_summary.json').read_text())
        for a in data['algorithms']:
            g=graph(a,True)
            self.assertEqual(sum(n['kind']=='llm' for n in g['nodes']),len(a['calls']))
            self.assertEqual(sum(n['kind']=='embedding' for n in g['nodes']),len(a['embedding_kernels']))
            self.assertTrue(all('start' not in n and 'end' not in n for n in g['nodes']))


if __name__=='__main__':unittest.main()
