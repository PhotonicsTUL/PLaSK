#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.slab import BesselCyl

from optical.effective import EffectiveFrequencyCyl


@material.simple()
class Core(material.Material):
    @staticmethod
    def Nr():
        return 3.5


class Disk(unittest.TestCase):

    def setUp(self):
        
        R = 20.
        N = 7
        
        disk = geometry.Rectangle(5., 0.5, 'Core')
        side = geometry.Rectangle(R-5., 0.5, 'air')
        layer = geometry.Shelf()
        layer.append(disk)
        layer.append(side)
        above = geometry.Rectangle(R, 2.0, 'air')
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
        self.solver.size = N
        
        self.layer = 0

    def testIntegrals(self):
        self.solver.wavelength = 1500
        self.solver.m = 1
        set_printoptions(precision=6, linewidth=180, suppress=True)
        ieps_minus = self.solver.ieps_minus(self.layer)
        print "\nieps minus ="
        print real(ieps_minus)
        print "\nieps plus ="
        print real(self.solver.ieps_plus(self.layer))
        print "\neps minus ="
        print real(self.solver.eps_minus(self.layer))
        print "\neps plus ="
        print real(self.solver.eps_plus(self.layer))
        print "\ndeps minus ="
        print real(self.solver.deps_minus(self.layer))
        print "\ndeps plus ="
        print real(self.solver.deps_plus(self.layer))
        print

    def testMatrices(self):
        self.solver.wavelength = 1500
        self.solver.m = 1
        RE, RH = self.solver.get_matrices(self.layer)
        set_printoptions(precision=1, linewidth=240, suppress=True)
        print "\nRE = "
        print abs(RE)
        print "\nRH = "
        print abs(RH)
        print "\nQE = "
        print abs(dot(RH,RE))
        print
        
    def testComputations(self):
        
        lams = linspace(1000, 5000, 700)
        dets = self.solver.get_determinant(lam=lams, dispersive=False)
        plot(lams, abs(dets))
        yscale('log')
        show()
        
        #m = self.solver.find_mode(1550)
        #self.assertEqual( m, 0 )
        #self.assertEqual( len(self.solver.modes), 1 )
        #self.assertAlmostEqual( self.solver.modes[m].lam, 1561.1-123.0j, 0 )
