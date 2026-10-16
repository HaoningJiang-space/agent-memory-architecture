"""Independent arithmetic, capacity and counterfactual-bound checks."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from architecture.explore import break_even, lifecycle, scenario_costs, profile
from architecture.model import candidates, decoder, gemm, GIB
from workload_accounting import request_work

ROOT = Path(__file__).resolve().parent


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT/'architecture/spec.json').read_text())
        cls.hs, cls.rejected = candidates(cls.spec)
        cls.h = next(h for h in cls.hs if h.name=='HBM3E4_a16_128x128_s32')
        cls.trace = json.loads((ROOT/'evidence/phase2/workload_accounting.json').read_text())

    def test_public_units_and_budget(self):
        # 4*24 decimal GB are not 96 GiB. SRAM 30 W/GB is not 30 kW/GB.
        self.assertEqual(self.h.capacity_bytes,96_000_000_000)
        self.assertLess(self.h.static_w,40)
        self.assertEqual(len(self.hs)+len(self.rejected),36)
        for h in self.hs:
            self.assertLessEqual(h.cost_units,self.spec['budget']['cost_units'])
            self.assertLessEqual(h.peak_power_w,self.spec['budget']['power_w'])

    def test_independent_projection_and_first_output_count(self):
        toy = dict(num_hidden_layers=2,hidden_size=4,num_attention_heads=2,
                   num_key_value_heads=1,head_dim=2,intermediate_size=6,vocab_size=10)
        r = decoder(self.h,toy,3,2)
        self.assertEqual(r['prefill']['flops'],1712)
        self.assertEqual(r['decode']['flops'],688)
        self.assertEqual(r['kv_bytes'],64)
        single = decoder(self.h,toy,3,1)
        self.assertEqual(single['decode']['seconds'],0)
        self.assertEqual(single['decode']['read_bytes'],0)

    def test_real_config_flops_agree_with_existing_independent_accountant(self):
        for group in self.trace['groups'].values():
            for c in group['calls']:
                cfg = self.trace['models'][c['model']]['config']
                actual = decoder(self.h,cfg,c['prompt_tokens'],c['completion_tokens'])
                expected = request_work(cfg,c['prompt_tokens'],c['completion_tokens'])
                self.assertEqual(actual['prefill']['flops'],expected['prefill_logical_flops'])
                # Central context approximation can differ by half a token.
                diff = abs(actual['decode']['flops']-expected['decode_logical_flops'])
                self.assertLessEqual(diff,2*cfg['num_hidden_layers']*cfg['num_attention_heads']*
                                     cfg['head_dim']*(c['completion_tokens']-1))

    def test_finite_tile_footprint_and_no_free_weight_stream(self):
        for m,k,n in [(1,4096,12288),(20796,5120,25600),(9,17,21)]:
            r = gemm(self.h,m,k,n)
            self.assertLessEqual(r['buffer_bytes'],self.h.sram_mib*2**20/self.h.arrays)
            self.assertEqual(r['flops'],2*m*k*n)
            self.assertGreaterEqual(r['read_bytes'],2*k*n+2*m*k)
            self.assertGreaterEqual(r['seconds'],r['compute_seconds'])
            self.assertGreaterEqual(r['seconds'],r['memory_seconds'])

    def test_relabeling_pool_cannot_create_speedup(self):
        a = gemm(self.h,512,4096,12288)
        b = gemm(replace(self.h,name='memory_specialized_engine'),512,4096,12288)
        self.assertEqual(a,b)

    def test_identical_length_no_fake_query_saving_and_offline_unload(self):
        emb = json.loads((ROOT/'evidence/architecture/embedding_shapes.json').read_text())
        capacity = json.loads((ROOT/'evidence/calibration/model_capacity.json').read_text())
        weights = {r['model']:r['weight_bytes'] for r in capacity['models']}
        groups = profile(self.h,self.trace,emb)
        raw = scenario_costs(self.h,self.trace,groups,weights,None,'online')
        same = scenario_costs(self.h,self.trace,groups,weights,100000,'online')
        self.assertEqual(raw['query'],same['query'])
        self.assertLessEqual(same['max_active_slots'],raw['max_active_slots'])
        off = scenario_costs(self.h,self.trace,groups,weights,4096,'offline')
        self.assertEqual(off['extra_online_weight_bytes'],0)
        self.assertEqual(off['overlap_construction_kv_bytes'],0)

    def test_break_even_and_maintenance_can_remove_it(self):
        self.assertEqual(break_even(100,2),50)
        self.assertIsNone(break_even(100,2,2))
        self.assertIsNone(break_even(100,-2))
        self.assertEqual(lifecycle(100,1,50),150)
        self.assertEqual(lifecycle(0,3,50),150)


if __name__=='__main__':
    unittest.main()
