import unittest
import json
from pathlib import Path
import numpy as np
from architecture.hetero_boundary import traffic, n_shard_reference, arrays, services, required_bandwidth, projection_streams


class HeteroBoundaryTest(unittest.TestCase):
    def test_partition_executes_same_matrix_and_keeps_tail(self):
        rng = np.random.default_rng(17)
        x, w = rng.normal(size=(7,19)), rng.normal(size=(19,31))
        np.testing.assert_allclose(n_shard_reference(x,w,8), x@w, atol=1e-12)

    def test_count_input_fanout_and_compression_crossover(self):
        a = traffic(4,64,128,.8,8)
        self.assertEqual(a['activation_boundary_bytes'], 2*4*(8*64+128))
        self.assertEqual(a['weight_only_intensity'], 5)
        self.assertAlmostEqual(a['traffic_crossover_m'], .8*traffic(4,64,128,1,8)['traffic_crossover_m'])

    def test_expanding_escape_blocks_are_not_free(self):
        row=traffic(1,64,128,1.125,8)
        self.assertEqual(row['encoded_weight_bytes'],1.125*row['raw_weight_bytes'])

    def test_identical_supply_never_rewards_partition_label(self):
        rows = [dict(m=m,k=512,n=768,count=1) for m in (1,4,29,512,20000)]
        for ratio in (1.,.8):
            data = arrays(rows,ratio,16)
            for fraction in (.1,.25,.5):
                u,h,_=services(data,1e12,1e10,fraction,1,4,overhead=0)
                self.assertTrue(np.all(h>=u))

    def test_sink_cap_can_block_any_local_bandwidth(self):
        data=arrays([dict(m=1,k=4096,n=4096,count=10)],.8,16)
        r=required_bandwidth(data,1e15,1e12,.1,1)
        self.assertIsNone(r['required_local_bandwidth_ratio'])

    def test_threshold_has_both_sides(self):
        data=arrays([dict(m=1,k=4096,n=4096,count=10)],.8,16)
        r=required_bandwidth(data,1e15,1e12,.1,4)
        gamma=r['required_local_bandwidth_ratio']
        self.assertIsNotNone(gamma)
        for scale,winning in [(0.99,False),(1.01,True)]:
            u,h,_=services(data,1e15,1e12,.1,gamma*scale,4)
            self.assertEqual(bool(h.sum()<u.sum()),winning)

    def test_conservation_and_legitimate_large_batch(self):
        _, streams=projection_streams()
        self.assertEqual(len(streams),6)
        family=json.loads((Path(__file__).parent/'evidence/architecture/family_summary.json').read_text())
        expected={a['algorithm']:len(a['calls'])+len(a['embedding_kernels']) for a in family['algorithms']}
        for stream in streams:
            self.assertEqual(stream['neural_nodes'],expected[stream['graph'].split('_')[0]])
        unlimited=[s for s in streams if s['graph'].startswith('lightmem') and s['maintenance_cap'] is None]
        self.assertTrue(all(any(r['phase']=='decode' and r['m']==29 for r in s['rows']) for s in unlimited))


if __name__=='__main__':
    unittest.main()
