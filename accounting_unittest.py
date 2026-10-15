"""Independent small-matrix and first-token checks for logical work accounting."""
import json
from pathlib import Path
import unittest

from workload_accounting import decoder_shape, request_work


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.toy = dict(num_hidden_layers=2, hidden_size=4, num_attention_heads=2,
                        head_dim=2, num_key_value_heads=1, intermediate_size=6, vocab_size=10)

    def test_small_matrices(self):
        # Per layer: Q/O 2*(4x4), K/V 2*(4x2), MLP 3*(4x6).
        # Two layers = 240 parameters; LM head 4x10 = 40 parameters.
        row = request_work(self.toy, prompt=3, completion=2)
        self.assertEqual(row['prefill_logical_flops'], 1712)
        self.assertEqual(row['decode_logical_flops'], 688)
        self.assertEqual(row['decode_weight_stream_bytes_B1'], 560)
        self.assertEqual(row['decode_kv_logical_read_bytes'], 64)
        self.assertEqual(row['peak_request_kv_bytes'], 64)

    def test_one_output_requires_no_decode_forward(self):
        row = request_work(self.toy, prompt=3, completion=1)
        self.assertEqual(row['decode_steps'], 0)
        self.assertEqual(row['decode_logical_flops'], 0)
        self.assertEqual(row['decode_kv_logical_read_bytes'], 0)
        self.assertEqual(row['kv_logical_write_bytes'], 48)

    def test_qwen_query_width_is_not_hidden_size(self):
        root = Path(__file__).resolve().parent
        models = json.loads((root/'evidence/phase2/workload_accounting.json').read_text())['models']
        config = models['Qwen/Qwen3-32B']['config']
        self.assertEqual(config['hidden_size'], 5120)
        shape = decoder_shape(config)
        self.assertEqual(shape['query_width'], 8192)
        self.assertEqual(shape['kv_bytes_per_token'], 262144)


if __name__ == '__main__':
    unittest.main()
