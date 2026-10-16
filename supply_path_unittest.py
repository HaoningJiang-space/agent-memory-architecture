"""Supply invariants independent of the exploratory winner selection."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from architecture import model as base
from architecture import mechanism as mech
from architecture.supply_path import (Route, LocalPath, provision, score, linear,
                                     fraction_options, cold_fill, backing_route, LIMITS)


class SupplyPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=json.loads(Path('architecture/spec.json').read_text())
        cls.hw=next(h for h in base.candidates(cls.spec)[0] if h.name=='HBM3E4_a16_128x128_s128')
        cls.hw=replace(cls.hw,fill_mode='continuous_k')
        cls.kh,_,_=mech.provision(cls.hw,mech.KShard(groups=2),cls.spec)

    def test_route_latency_window_and_bank_independence(self):
        route=Route(4,1000,10000,.1,1,10,interface_efficiency=1)
        self.assertEqual(route.capacities(),dict(bank=4000,interface=10000,window=400))
        self.assertAlmostEqual(route.service(800),2.1)
        self.assertEqual(route.service(0),0)
        self.assertAlmostEqual(replace(route,interface_bytes_s=100).service(800),8.1)
        large=replace(route,payload_bytes=1000)
        self.assertAlmostEqual(large.service(800),.3)
        self.assertAlmostEqual(replace(large,bank_skew=2).service(800),.5)
        # More interface pins cannot bypass the in-flight request ceiling.
        self.assertEqual(replace(route,interface_bytes_s=1e20).service(800),route.service(800))

    def test_backing_resident_conservation_and_no_free_bandwidth(self):
        row=mech.sharded(self.kh,mech.KShard(groups=2),1,4096,14336)
        row=dict(row,output_columns=14336,counted_k_device=True)
        path=LocalPath();a=score(self.kh,row,path,.25,True)
        self.assertAlmostEqual(a['backing_read_bytes']+a['hb_read_bytes'],row['read_bytes'])
        self.assertAlmostEqual(a['backing_write_bytes'],row['write_bytes'])
        self.assertAlmostEqual(a['local_sram_bytes'],2*a['hb_read_bytes'])
        self.assertGreaterEqual(a['seconds']-a['reduction_seconds'],backing_route(self.kh).service(a['backing_read_bytes']+a['backing_write_bytes']))
        self.assertAlmostEqual(sum(a['limited_by_'+k+'_seconds'] for k in LIMITS)+a['limited_by_reduction_seconds'],a['seconds'])
        off=score(self.kh,row,path,0,True)
        normal=score(self.kh,row)
        self.assertEqual(off['seconds'],normal['seconds'])
        self.assertEqual(off['hb_read_bytes'],0)
        self.assertEqual(off['energy_j'],normal['energy_j'])

    def test_selected_mapping_traffic_classes_conserve_routes(self):
        for hw,counted,path,fraction in [(self.hw,False,None,0),
                                         (self.kh,True,None,0),
                                         (self.kh,True,LocalPath(),.15)]:
            for m,k,n in [(1,4096,14336),(237,4096,14336)]:
                row=linear(hw,m,k,n,counted,path,fraction)
                reads=sum(row[key] for key in ('weight_read_bytes','kv_read_bytes',
                                                'activation_read_bytes','partial_read_bytes'))
                writes=row['activation_write_bytes']+row['partial_write_bytes']
                self.assertAlmostEqual(reads,row['backing_read_bytes']+row['hb_read_bytes'])
                self.assertAlmostEqual(writes,row['backing_write_bytes'])
        # Attention carries KV and activation labels from the frozen mechanism.
        source=mech.attention(self.hw,239,64,8,128,decode=True)
        row=score(self.hw,source)
        self.assertEqual(row['kv_read_bytes']+row['activation_read_bytes'],row['backing_read_bytes'])
        self.assertEqual(row['activation_write_bytes'],row['backing_write_bytes'])

    def test_capacity_cost_and_buffer_ports_are_charged(self):
        p=LocalPath(banks=32,gb_s_per_bank=51.2,ports_per_bank=4)
        hw,charge,bad=provision(self.kh,p,self.spec)
        self.assertAlmostEqual(charge['controller_mm2'],32*6.76*.402)
        self.assertEqual(p.buffer_bytes,2*32*4*4096)
        self.assertGreater(hw.area_mm2,self.kh.area_mm2)
        self.assertGreater(hw.cost_units,self.kh.cost_units+100)
        self.assertGreater(hw.peak_power_w,self.kh.peak_power_w)
        self.assertLess(hw.frequency,self.kh.frequency)
        _,few,_=provision(self.kh,replace(p,ports_per_bank=1),self.spec)
        self.assertLess(few['local_sram_mm2'],charge['local_sram_mm2'])
        options=fraction_options(65_524_246_528,p)
        self.assertAlmostEqual(max(options),10e9/65_524_246_528)
        self.assertLess(max(options),.16)
        fill=cold_fill(hw,p,65_524_246_528,max(options))
        self.assertEqual(fill['bytes'],10e9)
        self.assertGreater(fill['seconds'],0)
        self.assertGreater(fill['energy_j'],0)

    def test_port_window_and_skew_have_real_service_cost(self):
        row=mech.sharded(self.kh,mech.KShard(groups=2),1,4096,14336)
        row=dict(row,output_columns=14336,counted_k_device=True)
        p=LocalPath(banks=64,gb_s_per_bank=51.2,ports_per_bank=4)
        fast=score(self.kh,row,p,1,True)
        slow=score(self.kh,row,replace(p,ports_per_bank=1),1,True)
        self.assertGreater(slow['seconds'],fast['seconds'])
        window=score(self.kh,row,replace(p,buffer_bytes_per_port=64),1,True)
        self.assertGreater(window['seconds'],fast['seconds'])
        skew=score(self.kh,row,replace(p,bank_skew=4),1,True)
        self.assertGreater(skew['seconds'],fast['seconds'])
        self.assertEqual(skew['hb_read_bytes'],fast['hb_read_bytes'])

    def test_os_ws_optimizer_and_k_choice_preserve_computation(self):
        for m,k,n in [(1,17,19),(237,4096,14336)]:
            row=linear(self.hw,m,k,n)
            self.assertEqual(row['flops'],2*m*k*n)
            self.assertEqual(row['hb_read_bytes'],0)
            self.assertGreater(row['seconds'],0)
        p=LocalPath()
        # Even with an HB copy, known-token matrix processing remains on the
        # ordinary path; we do not assume local low-M routing preserves reuse.
        plain=linear(self.kh,237,4096,14336,True)
        extra=linear(self.kh,237,4096,14336,True,p,.15)
        self.assertEqual(plain['seconds'],extra['seconds'])
        self.assertEqual(extra['hb_read_bytes'],0)
        forced_slow=replace(p,bank_skew=100)
        fallback=linear(self.kh,1,4096,14336,True,forced_slow,.15)
        plain=linear(self.kh,1,4096,14336,True)
        self.assertEqual(fallback['seconds'],plain['seconds'])
        self.assertEqual(fallback['hb_read_bytes'],0)


if __name__=='__main__':unittest.main()
