"""Independent small counterexamples for resource-envelope interpretation."""
import unittest
from architecture.service_envelope import pooled_bound, bundle_frontier


class ServiceEnvelopeTest(unittest.TestCase):
    def test_pool_covers_every_mixture(self):
        rows = [dict(domain='N', u_seconds=2, h_seconds=5),
                dict(domain='N', u_seconds=1, h_seconds=4),
                dict(domain='C', u_seconds=3, h_seconds=6)]
        bound = pooled_bound(rows)
        self.assertAlmostEqual(bound['sum_alpha'], .9)
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    self.assertLessEqual(2*a+b+3*c, .9*max(5*a+4*b, 6*c)+1e-10)

    def test_individual_dominance_is_not_pool_dominance(self):
        rows = [dict(domain='N', u_seconds=3, h_seconds=4),
                dict(domain='C', u_seconds=3, h_seconds=4)]
        self.assertFalse(pooled_bound(rows)['pooled_capacity_covers_split'])
        # Individually faster U needs 6 s, parallel H needs only 4 s.
        self.assertGreater(sum(r['u_seconds'] for r in rows), max(r['h_seconds'] for r in rows))

    def test_bad_time_is_not_free_work(self):
        with self.assertRaises(ValueError):
            pooled_bound([dict(domain='N', u_seconds=0, h_seconds=2)])

    def test_bundle_optimum_at_crossing(self):
        rows = [dict(stage='construction', domain='N', u_seconds=4, h_seconds=8),
                dict(stage='query_mixed', domain='C', u_seconds=3, h_seconds=6)]
        result = bundle_frontier(rows)
        # Balanced x=8/6: U=8 and H=8; each endpoint is only 0.5.
        self.assertAlmostEqual(result['max_ratio'], 1)
        self.assertAlmostEqual(result['extrema'][2]['query_bundles_per_history'], 4/3)


if __name__ == '__main__':
    unittest.main()
