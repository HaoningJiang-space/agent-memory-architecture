import unittest
import numpy as np
from architecture.weight_codec_screen import encode,decode,window,address_table_bytes,ALIGN


class WeightCodecTest(unittest.TestCase):
    def test_all_bf16_patterns_and_exact_special_values(self):
        all_bits=np.arange(65536,dtype=np.uint16)
        for block in np.split(all_bits,256):
            blob=encode(block,window(block))
            np.testing.assert_array_equal(decode(blob),block)
            self.assertEqual(len(blob)%ALIGN,0)

    def test_common_exponents_and_escape_mix(self):
        rng=np.random.default_rng(42)
        exponent=np.repeat(np.arange(117,124,dtype=np.uint16),256)
        bits=(exponent<<7)|rng.integers(0,128,len(exponent),dtype=np.uint16)|rng.integers(0,2,len(exponent),dtype=np.uint16)<<15
        bits[::31]=np.array([0x7fc1],dtype=np.uint16)
        blob=encode(bits,117)
        self.assertEqual(blob[1],1)
        np.testing.assert_array_equal(decode(blob),bits)
        self.assertLess(len(blob),bits.nbytes)

    def test_random_access_index_includes_header_and_alignment(self):
        self.assertEqual(address_table_bytes(96), 832)
        self.assertEqual(address_table_bytes(1), 64)
        with self.assertRaises(ValueError):
            address_table_bytes(0)

    def test_raw_fallback_counts_header_and_alignment(self):
        words=np.arange(256,dtype=np.uint16)
        blob=encode(words,100)
        self.assertEqual(blob[1],0)
        self.assertEqual(len(blob),576)
        np.testing.assert_array_equal(decode(blob),words)


if __name__=='__main__':unittest.main()
