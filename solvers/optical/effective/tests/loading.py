#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from plask import *

@material.simple
class LowContrastMaterial(material.Material):
    def Nr(self, wl, T): return 1.3

class EffectiveIndex2D_Test(unittest.TestCase):

    def setUp(self):
        manager = Manager()
        manager.read("""
            <solvers>
                <optical lib="effective" solver="EffectiveIndex2D" name="eff">
                    <polarization>TM</polarization>
                </optical>
            </solvers>
        """)
        self.solver = manager.slv.eff

    def testLoadConfigurations(self):
        self.assertEqual( self.solver.polarization, "TM" )
