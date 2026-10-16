"""Independent checks of the causal controls used in the research note."""
import unittest

from architecture.capacity_latency_bounds import capacity_batch, stream_step_floor


class ConstraintTests(unittest.TestCase):
    def test_shared_prefix_is_counted_once(self):
        # 100 bytes: 20 shared and four private suffixes of 20 bytes.
        self.assertEqual(capacity_batch(100, 2, 10, 10), 4)
        self.assertEqual(capacity_batch(100, 2, 0, 20), 2)

    def test_prefix_that_does_not_fit_cannot_admit_requests(self):
        self.assertEqual(capacity_batch(19, 2, 10, 1), 0)
        self.assertEqual(capacity_batch(20, 2, 10, 1), 0)

    def test_batch_amortizes_weights_not_each_request_dependency(self):
        # Read 100 bytes at 1000 B/s; four outputs, each still waits 0.1 s.
        one = stream_step_floor(100, 1, 0, 1000)
        four = stream_step_floor(100, 4, 0, 1000)
        self.assertAlmostEqual(one, 0.1)
        self.assertEqual(four, one)
        self.assertAlmostEqual(4 / four, 40)

    def test_kv_can_only_raise_streaming_floor(self):
        # 100 weight bytes plus four 25-byte KV streams = 200 bytes.
        self.assertAlmostEqual(stream_step_floor(100, 4, 25, 1000), 0.2)
        self.assertGreater(stream_step_floor(100, 4, 25, 1000),
                           stream_step_floor(100, 4, 0, 1000))


if __name__ == '__main__':
    unittest.main()
