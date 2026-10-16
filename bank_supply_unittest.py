import unittest

from architecture.bank_supply import Config, decode_cycles, simulate
from architecture.supply_resource_bounds import max_m


class BankSupplyTests(unittest.TestCase):
    def test_single_bank_independent_recurrence(self):
        blocks = [[384, 360, 0, 1], [448, 410, 50, 1], [576, 520, 0, 0]]
        cfg = Config(input_credits=1, output_credits=1, m=8, index_header_bytes=0)
        read_end = decode_end = consume_end = 0
        for block in blocks:
            read_end = decode_end + (block[0]+8)/cfg.bank_bytes_cycle
            decode_end = max(read_end, consume_end) + decode_cycles(block, cfg)
            consume_end = decode_end + 256*cfg.m/cfg.macs_per_bank
        got = simulate([blocks], cfg)
        self.assertAlmostEqual(got['cycles'], consume_end)
        self.assertEqual(got['bank_bytes'], sum(b[0]+8 for b in blocks))

    def test_uniform_pool_has_no_free_gain(self):
        banks = [[[384, 360, 0, 1]]*32 for _ in range(4)]
        local = simulate(banks, Config())
        ideal = simulate(banks, Config(group=4))
        self.assertAlmostEqual(local['cycles'], ideal['cycles'])

    def test_routing_counts_both_sides(self):
        banks = [[[384, 360, 0, 1]]*5 for _ in range(4)]
        ideal = simulate(banks, Config(group=4))
        routed = simulate(banks, Config(group=4, link_bytes_cycle=16, link_start_cycles=2))
        self.assertEqual(routed['crossbar_input_bytes'], 20*(384+8))
        self.assertEqual(routed['crossbar_output_bytes'], 20*(512+8))
        self.assertEqual(routed['input_fifo_provision_bytes'], 4*2*(576+8))
        self.assertGreater(routed['cycles'], ideal['cycles'])
        self.assertEqual(routed['weight_consumer_macs'], 20*256*4)

    def test_backpressure_and_conservation(self):
        banks = [[[384, 360, 0, 1]]*20 for _ in range(4)]
        result = simulate(banks, Config(m=29, macs_per_bank=8, bank_bytes_cycle=1000,
                                       input_credits=1, output_credits=1, group=4))
        self.assertEqual(result['blocks'], 80)
        self.assertEqual(result['max_input_credits'], 1)
        self.assertEqual(result['max_output_credits'], 1)
        self.assertGreater(result['input_backpressure_cycles'], 0)
        self.assertGreater(result['output_backpressure_cycles'], 0)

    def test_imbalance_positive_control(self):
        banks = [[[384, 360, 0, 1]]*32 for _ in range(4)]
        banks[0] = [[512, 480, 120, 1]]*32
        cfg = dict(bank_bytes_cycle=1000, macs_per_bank=256, m=1, input_credits=4, output_credits=4)
        local = simulate(banks, Config(**cfg))
        ideal = simulate(banks, Config(**cfg, group=4))
        self.assertGreater(local['cycles']/ideal['cycles'], 1.5)

    def test_reuse_does_not_repeat_decode(self):
        banks = [[[384, 360, 0, 1]]*8 for _ in range(4)]
        a = simulate(banks, Config(m=1))
        b = simulate(banks, Config(m=29))
        self.assertEqual(a['bank_bytes'], b['bank_bytes'])
        self.assertEqual(a['decoder_work_cycles'], b['decoder_work_cycles'])
        self.assertEqual(b['weight_consumer_macs'], a['weight_consumer_macs']*29)

    def test_invalid_group(self):
        with self.assertRaises(ValueError):
            simulate([[[384, 360, 0, 1]]], Config(group=0))

    def test_interface_mac_matching(self):
        self.assertAlmostEqual(max_m(32, 51.2e9, .8), 1)
        self.assertAlmostEqual(max_m(64, 12.8e9, .8), 8)
        self.assertAlmostEqual(max_m(64, 12.8e9, .8), max_m(64, 12.8e9, 1)*.8)

    def test_pipelined_link_not_per_packet_throughput_tax(self):
        banks = [[[384, 360, 0, 1]]*32 for _ in range(4)]
        a = simulate(banks, Config(group=4, link_bytes_cycle=16, link_start_cycles=0,
                                   input_credits=8, output_credits=8))
        b = simulate(banks, Config(group=4, link_bytes_cycle=16, link_start_cycles=2,
                                   input_credits=8, output_credits=8))
        # Long link transfers dominate. Pipelined 2-cycle propagation does not
        # add 2 cycles to the serialization time of all 256 transfers.
        self.assertLess(b['cycles']-a['cycles'], 20)


if __name__ == '__main__':
    unittest.main()
