import unittest
from architecture.collect_family import overlap, metrics


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


if __name__ == '__main__':
    unittest.main()
