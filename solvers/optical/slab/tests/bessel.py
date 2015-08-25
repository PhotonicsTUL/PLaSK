#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.slab import BesselCyl

from optical.effective import EffectiveFrequencyCyl


class Disk(unittest.TestCase):

    def setUp(self):
        disk = geometry.Rectangle(5., 0.5, 'GaAs')
        side = geometry.Rectangle(15., 0.5, 'air')
        layer = geometry.Shelf()
        layer.append(disk)
        layer.append(side)
        above = geometry.Rectangle(20., 2.0, 'air')
        stack = geometry.Stack2D()
        stack.prepend(above)
        stack.prepend(layer)
        stack.prepend(above)
        self.geometry = geometry.Cylindrical2D(stack)
        
        #self.solver = EffectiveFrequencyCyl('EFM')
        #self.solver.geometry = self.geometry
        #self.solver.lam0 = 1500.
        #self.solver.vat = 0.
        
        self.solver = BesselCyl('Bessel')
        self.solver.geometry = self.geometry
        self.solver.set_interface(stack)
        self.solver.size = 7

    def testIntegrals(self):
        self.solver.wavelength = 1500
        self.solver.m = 1
        set_printoptions(precision=6, linewidth=180, suppress=True)
        print real(self.solver.ieps_minus(0))
        print
        print real(self.solver.ieps_plus(0))
        print
        print real(self.solver.eps_minus(0))
        print
        print real(self.solver.eps_plus(0))
        print
        print real(self.solver.deps_minus(0))
        print
        print real(self.solver.deps_plus(0))
        print
        #print self.solver.ieps_minus(1)

    #def testComputations(self):
        
        #lams = linspace(1000, 5000, 700)
        #dets = self.solver.get_determinant(lam=lams, dispersive=False)
        #plot(lams, abs(dets))
        #yscale('log')
        #show()
        
        ##m = self.solver.find_mode(1550)
        ##self.assertEqual( m, 0 )
        ##self.assertEqual( len(self.solver.modes), 1 )
        ##self.assertAlmostEqual( self.solver.modes[m].lam, 1561.1-123.0j, 0 )
