"""Focused independent checks for residency, handoff and resource legality."""
import unittest

from architecture.call_placement import (M8, M32, EMB, capacities, hardware,
    replay, resource_bill)
from architecture.explore import load


class PlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.trace, _, capacity, _ = load()
        cls.weights = {r['model']: r['weight_bytes'] for r in capacity['models']}
        cls.hw = hardware(cls.spec, 'HBM3E', 4, 16, 128)
        cls.devices = {'C': cls.hw, 'N': hardware(cls.spec, 'HBM3E', 1, 8, 32)}

    def dag(self, calls):
        return {'nodes': calls, 'query_ends': [calls[-1]['id']]}

    def call(self, identity, deps=(), name=M8, length=16, output=3):
        return dict(id=identity, deps=list(deps), kind='llm', model=name,
                    L=length, O=output, stage='query_0')

    def test_split_capacity_and_one_token_completion(self):
        cfg = self.trace['models'][M8]['config']
        calls = [self.call('a', length=9, output=1), self.call('b', length=5, output=4)]
        r = capacities(cfg, calls, 100, 200)
        per_token = 36 * 8 * 128 * 2 * 2
        self.assertEqual(r['kv_transfer_bytes'], per_token * 9)
        self.assertEqual(r['PD_C'], 100 + per_token*2*9 + 2*2*9*(4096+2*12288))
        self.assertGreaterEqual(r['PD_N'], 200 + per_token*9)

    def test_pd_duplicates_full_checkpoint(self):
        dag = self.dag([self.call('a')])
        whole = replay(dag, self.devices, {'C':[M32, EMB], 'N':[M8]},
                       self.trace, self.weights, 'whole')
        pd = replay(dag, self.devices, {'C':[M32, EMB, M8], 'N':[M8]},
                    self.trace, self.weights, 'pd')
        self.assertEqual(pd['initial_load_bytes'] - whole['initial_load_bytes'], self.weights[M8])
        self.assertAlmostEqual(pd['initial_load_seconds'] - whole['initial_load_seconds'], self.weights[M8]/32e9)
        self.assertEqual(pd['model_copies'][M8], 2)
        self.assertEqual(pd['batches'][0]['kv_bytes'], 16*147456)

    def test_all_first_tokens_need_no_decode_handoff(self):
        r = replay(self.dag([self.call('a', output=1)]), self.devices,
                   {'C':[M8], 'N':[M8]}, self.trace, self.weights, 'pd')
        self.assertEqual(r['batches'][0]['route'], 'C')
        self.assertEqual(r['batches'][0]['kv_bytes'], 0)
        self.assertEqual(r['reservations']['N'], [])

    def test_dependency_and_actual_shrinking_batch(self):
        calls = [self.call('a', output=2), self.call('b', output=4),
                 self.call('c', ('a',), output=2)]
        r = replay(self.dag(calls), {'C':self.hw}, {'C':[M8]}, self.trace, self.weights, 'unified')
        self.assertEqual(r['batches'][0]['segment_batches'], [(2,1),(1,2)])
        self.assertLess(r['completions']['a'], r['completions']['b'])
        self.assertGreaterEqual(r['starts']['c'], r['completions']['b'])

    def test_unified_eviction_reloads_returning_model(self):
        calls = [self.call('a'), self.call('b', ('a',), name=M32), self.call('c', ('b',))]
        r = replay(self.dag(calls), {'C':self.hw}, {'C':[M8,M32]}, self.trace,
                   self.weights, 'unified', active_only=True)
        self.assertEqual(r['initial_load_bytes'], 0)
        self.assertEqual(sum(b['reload_bytes'] for b in r['batches']), 2*self.weights[M8]+self.weights[M32])
        self.assertLess(r['peak_bytes']['C'], self.weights[M8]+self.weights[M32])

    def test_busy_domain_does_not_block_other_ready_model(self):
        # a finishes before the long b in the same N batch. Its dependent c
        # waits for N, but a later ready 32B call d must use the idle C domain.
        calls = [self.call('a', output=2), self.call('b', output=100),
                 self.call('c', ('a',)), self.call('d', ('a',), name=M32)]
        r = replay(self.dag(calls), self.devices, {'C':[M32], 'N':[M8]},
                   self.trace, self.weights, 'whole')
        self.assertLess(r['starts']['d'], r['completions']['b'])

    def test_zero_transfer_control_preserves_bytes_and_energy_charge(self):
        dag = self.dag([self.call('a', length=128)])
        args = (dag, self.devices, {'C':[M8], 'N':[M8]}, self.trace, self.weights, 'pd')
        a = replay(*args)
        b = replay(*args, free_kv_time=True)
        self.assertEqual(a['batches'][0]['kv_bytes'], b['batches'][0]['kv_bytes'])
        self.assertAlmostEqual(a['dynamic_j'], b['dynamic_j'])
        self.assertGreater(a['seconds'], b['seconds'])

    def test_residency_and_link_area_are_not_free(self):
        with self.assertRaises(ValueError):
            replay(self.dag([self.call('a')]), {'C':self.devices['N']},
                   {'C':[M32]}, self.trace, self.weights, 'unified')
        devices = dict(C=hardware(self.spec,'LPDDR5X',8,16,64), N=self.devices['N'])
        self.assertEqual(resource_bill(devices,self.spec,8,8)['violations'], [])
        self.assertIn('area_mm2', resource_bill(devices,self.spec,16,8)['violations'])


if __name__ == '__main__':
    unittest.main()
