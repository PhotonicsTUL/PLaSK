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
        
        R = 19.
        N = 7
        
        self.f = f = 1
        
        disk = geometry.Rectangle(5./f, 0.5/f, 'Core')
        side = geometry.Rectangle(R-5./f, 0.5/f, 'air')
        layer = geometry.Shelf()
        layer.append(disk)
        layer.append(side)
        above = geometry.Rectangle(R/f, 2.0/f, 'air')
        stack = geometry.Stack2D()
        stack.prepend(above)
        stack.prepend(layer)
        stack.prepend(above)
        self.geometry = geometry.Cylindrical2D(stack)
        

        #self.solver = EffectiveFrequencyCyl('EFM')
        #self.solver.geometry = self.geometry
        #self.solver.lam0 = 1500./f
        #self.solver.vat = 0.
        
        self.solver = BesselCyl('Bessel')
        self.solver.geometry = self.geometry
        self.solver.set_interface(stack)
        self.solver.size = N
        self.solver.pml.dist = 20./f - R
        self.solver.lam0 = 1550/f
        
        self.layer = 0
        
        plot_geometry(self.geometry, fill=True, margin=0.1)

    def testIntegrals(self):
        try:
            self.solver.wavelength = 1500/self.f
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
            print real(self.solver.deps_pluself.Rs(self.layer))
            print
        except AttributeError:
            pass

    def testMatrices(self):
        try:
            self.solver.wavelength = 1500/self.f
            self.solver.m = 1
            RE, RH = self.solver.get_matrices(self.layer)
        except AttributeError:
            pass
        else:
            set_printoptions(precision=1, linewidth=240, suppress=True)
            print "\nRE = "
            print abs(RE)
            print "\nRH = "
            print abs(RH)
            print "\nQE = "
            print abs(dot(RH,RE))
            print

    def plot_determinant(self):
        lams = linspace(1000/self.f, 5000/self.f, 700)
        try:
            dets = self.solver.get_determinant(lam=lams, m=1, dispersive=False)
        except TypeError:
            dets = self.solver.get_determinant(lam=lams)
        figure()
        plot(lams, abs(dets))
        yscale('log')
        
    def testComputations(self):
        m = self.solver.find_mode(1550/self.f)
        self.assertEqual( m, 0 )
        self.assertEqual( len(self.solver.modes), 1 )
        #self.assertAlmostEqual( self.solver.modes[m].lam, 1561.1-123.0j, 0 )


if __name__ == "__main__":
    disk = Disk('plot_determinant')
    disk.setUp()

    disk.plot_determinant()
    #disk.plot_field()
    show()
    