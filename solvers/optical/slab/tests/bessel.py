#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh

from optical.slab import BesselCyl
from optical.effective import EffectiveFrequencyCyl


@material.simple()
class Core(material.Material):
    Nr = 3.5


class Disk(unittest.TestCase):

    def setUp(self):

        R = 9.
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

        self.solver.pml.dist = 10./f - R

        self.solver.pml.size = 2.
        self.solver.pml.factor = 1.-2.j
        self.solver.pml.shape = 0

        self.solver.lam0 = 1550/f

        self.layer = 0

    def plot_geometry(self):
        plot_geometry(self.geometry, fill=True, margin=0.1)

    def print_integrals(self):
        try:
            self.solver.wavelength = 1500/self.f
            self.solver.m = 1
            set_printoptions(precision=6, linewidth=180, suppress=True)
            epsVmm = self.solver.epsVmm(self.layer)
            print("\nepsVmm =")
            print(real(epsVmm))
            print("\nepsVpp =")
            print(real(self.solver.epsVpp(self.layer)))
            print("\nepsTmm =")
            print(real(self.solver.epsTmm(self.layer)))
            print("\nepsTpp =")
            print(real(self.solver.epsTpp(self.layer)))
            print("\nepsTmp =")
            print(real(self.solver.epsTmp(self.layer)))
            print("\nepsTpm =")
            print(real(self.solver.epsTpm(self.layer)))
            print("\nepsDm =")
            print(real(self.solver.epsDm(self.layer)))
            print("\nepsDp =")
            print(real(self.solver.epsDp(self.layer)))
            print
            print("\nmuVmm =")
            print(real(self.solver.muVmm()))
            print("\nmuVpp =")
            print(real(self.solver.muVpp()))
            print("\nmuTmm =")
            print(real(self.solver.muTmm()))
            print("\nmuTpp =")
            print(real(self.solver.muTpp()))
            print("\nmuTmp =")
            print(real(self.solver.muTmp()))
            print("\nmuTpm =")
            print(real(self.solver.muTpm()))
            print("\nmuDm =")
            print(real(self.solver.muDm()))
            print("\nmuDp =")
            print(real(self.solver.muDp()))
            print
        except AttributeError:
            import traceback
            traceback.print_exc()

    def print_matrices(self):
        try:
            self.solver.wavelength = 1500/self.f
            self.solver.m = 1
            RE, RH = self.solver.get_matrices(self.layer)
        except AttributeError:
            pass
        else:
            set_printoptions(precision=2, linewidth=240, suppress=True)
            print("\nRH = ")
            print(abs(RH))
            print("\nRE = ")
            print(abs(RE))
            print("\nQE = ")
            print(abs(dot(RH,RE)))
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

    try:
        function = sys.argv[1]
    except IndexError:
        function = 'setUp'

    disk = Disk(function)
    disk.setUp()

    if function != 'setUp':
        getattr(disk, function)()
    else:
        disk.print_integrals()
        disk.print_matrices()
        disk.plot_geometry()
        disk.plot_determinant()
        #disk.plot_field()
    show()
