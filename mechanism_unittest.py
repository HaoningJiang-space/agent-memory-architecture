import json
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
from architecture import model as base
from architecture.mechanism import KShard, provision, linear, attention, sharded, evaluate, execute_reference,TRAFFIC


class MechanismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads(Path('architecture/spec.json').read_text())
        cls.hw=next(h for h in base.candidates(cls.spec)[0] if h.name=='HBM3E4_a16_128x128_s128')
        cls.cfg=json.loads(Path('evidence/phase2/workload_accounting.json').read_text())['models']['Qwen/Qwen3-8B']['config']

    def test_frozen_baseline_and_traffic_conservation(self):
        for length,output in [(239,50),(1880,736)]:
            old=base.decoder(self.hw,self.cfg,length,output)
            new=evaluate(self.hw,self.cfg,length,output)
            for phase in ('total','prefill','decode'):
                for key,value in old[phase].items(): self.assertAlmostEqual(new[phase][key],value)
            for r in new['components'].values():
                self.assertAlmostEqual(sum(r[k] for k in TRAFFIC[:4]),r['read_bytes'])
                self.assertAlmostEqual(sum(r[k] for k in TRAFFIC[4:]),r['write_bytes'])

    def test_actual_partial_sum_schedule_with_tails(self):
        rng=np.random.default_rng(20261016)
        for k,n,g in [(17,19,4),(129,257,16),(3,7,8),(512,128,64)]:
            x=rng.normal(size=k).astype(np.float32);w=rng.normal(size=(k,n)).astype(np.float32)
            np.testing.assert_allclose(execute_reference(x,w,g,16),x@w,rtol=3e-4,atol=3e-5)

    def test_hand_count_and_resource_cost(self):
        h=replace(self.hw,arrays=1,rows=4,cols=2)
        s=KShard(groups=2,reduction_lanes=1,adder_latency_cycles=3,sync_cycles=8)
        modified,cost,rejected=provision(h,s,self.spec)
        r=sharded(modified,s,1,5,3)
        # Two output tiles, K partitions of length 3, 2x2 subarray.
        self.assertAlmostEqual(r['compute_seconds']*modified.frequency,2*(3+2+2-2))
        self.assertEqual(r['weight_read_bytes'],30)
        self.assertEqual(r['activation_read_bytes'],20)
        self.assertEqual(r['reduction_adds'],3)
        self.assertEqual(r['reduction_link_bytes'],24)
        self.assertAlmostEqual(sum(r['array_'+key+'_seconds'] for key in base.CYCLE_KEYS),r['compute_seconds'])
        self.assertGreater(cost['added_logic_area_mm2'],0)
        self.assertLess(modified.frequency,h.frequency)
        self.assertGreater(r['seconds'],r['reduction_seconds'])
        self.assertLessEqual(r['buffer_bytes'],modified.sram_mib*base.MIB)

    def test_sram_and_reduction_cannot_disappear(self):
        s=KShard(groups=16)
        h,_,_=provision(self.hw,s,self.spec)
        a=sharded(h,s,1,4096,14336)
        fast=sharded(replace(h,bandwidth=h.bandwidth*100),s,1,4096,14336)
        self.assertEqual(a['read_bytes'],fast['read_bytes'])
        self.assertGreaterEqual(fast['seconds'],fast['sram_seconds'])
        skew=sharded(h,replace(s,bank_skew=4),1,4096,14336)
        self.assertGreaterEqual(skew['seconds'],a['seconds'])
        self.assertEqual(skew['sram_bytes'],a['sram_bytes'])


if __name__=='__main__': unittest.main()
