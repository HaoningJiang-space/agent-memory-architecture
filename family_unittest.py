import unittest
from architecture.collect_family import overlap, metrics
from architecture.family_replay import segments, batched
from architecture import model as base, mechanism as mech, supply_path as supply
from architecture.explore import load


class FamilyEvidenceTest(unittest.TestCase):
    def test_parallel_work_is_not_wall_time_or_batch(self):
        result = overlap([dict(start=0, end=3), dict(start=1, end=2), dict(start=3, end=5)])
        self.assertEqual(result['api_concurrency_peak'], 2)
        self.assertEqual(result['api_interval_union_seconds'], 5)
        self.assertEqual(result['summed_api_seconds'], 6)
        self.assertEqual(overlap([])['api_interval_union_seconds'], 0)

    def test_engine_histogram_is_cumulative(self):
        text = ('# HELP ignored\nvllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="x"} 3.0\n'
                'vllm:iteration_tokens_total_bucket{engine="0",le="8.0",model_name="x"} 5.0\n')
        self.assertEqual(metrics(text), {'1.0': 3.0, '8.0': 5.0})

    def test_distinct_ragged_batch_preserves_every_decode_position(self):
        calls = [dict(id='a',L=20,O=1),dict(id='b',L=9,O=4),dict(id='c',L=15,O=7)]
        rows = segments(calls)
        self.assertEqual([(r['count'],r['batch'],r['padded_prompt']) for r in rows],[(3,2,15),(3,1,15)])
        self.assertEqual(sum(r['count']*r['batch'] for r in rows),9)
        with self.assertRaises(AssertionError):
            segments([calls[0],calls[0]])

    def test_single_request_batch_matches_existing_service(self):
        spec, trace, _, _, _ = load()
        hw = next(h for h in base.candidates(spec)[0] if h.name=='HBM3E4_a16_128x128_s128')
        hw, _, bad = mech.provision(hw,mech.KShard(groups=2),spec)
        self.assertFalse(bad)
        cfg = trace['models']['Qwen/Qwen3-8B']['config']
        for output in (1,8):
            expected = supply.evaluate(hw,cfg,29,output,True)
            actual = batched(hw,cfg,[dict(id='one',L=29,O=output)])
            for phase in ('prefill','decode','total'):
                for key in supply.FIELDS:
                    self.assertAlmostEqual(actual[phase][key],expected[phase][key],delta=max(1e-8,abs(expected[phase][key])*1e-12))


if __name__ == '__main__':
    unittest.main()
