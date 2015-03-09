#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from gain import Ferminew2D

@material.simple()
class Well(material.Material):
    def lattC(T=300., x='a'): return 5.5
    def nr(self, wl, T=300., n=0.): return 4.0
    def Eg(self, T=300., e=0., point='G'): return 0.5
    def VB(self, T=300., e=0., point='G', hole='H'): return 0.5
    def VB(self, T=300., e=0., point='G', hole='L'): return 0.5
    def CB(self, T=300., e=0., point='G'): return 1.0
    def Dso(self, T=300., e=0.): return 0.35
    def Me(self, T=300., e=0., point='G'): return 0.050
    def Mhh(self, T=300., e=0.): return 0.200
    def Mlh(self, T=300., e=0.): return 0.080

@material.simple()
class Barrier(material.Material):
    def Eg(self, T=300., e=0., point='G'): return 0.75
    def VB(self, T=300., e=0., point='G', hole='H'): return 0.45
    def VB(self, T=300., e=0., point='G', hole='L'): return 0.45
    def CB(self, T=300., e=0., point='G'): return 1.2
    def Dso(self, T=300., e=0.): return 0.35
    def Me(self, T=300., e=0., point='G'): return 0.050
    def Mhh(self, T=300., e=0.): return 0.200
    def Mlh(self, T=300., e=0.): return 0.080

@material.simple()
class Cladding(material.Material):
    def Eg(self, T=300., e=0., point='G'): return 0.9
    def VB(self, T=300., e=0., point='G', hole='H'): return 0.4
    def VB(self, T=300., e=0., point='G', hole='L'): return 0.4
    def CB(self, T=300., e=0., point='G'): return 1.3
    def Dso(self, T=300., e=0.): return 0.35
    def Me(self, T=300., e=0., point='G'): return 0.050
    def Mhh(self, T=300., e=0.): return 0.200
    def Mlh(self, T=300., e=0.): return 0.080

@material.simple()
class Substrate(material.Material):
    def Eg(self, T=300., e=0., point='G'): return 1.5

@material.simple()
class Cap(material.Material):
    def Eg(self, T=300., e=0., point='G'): return 1.5

class NewGainValues(unittest.TestCase):

    def setUp(self):
        self.solver = Ferminew2D("GAIN")
        plask.config.axes = 'xy'

        self.rect1 = geometry.Rectangle(10., 10., Substrate())
        self.rect1.role = 'substrate'
        self.rect2 = geometry.Rectangle(10., 0.01, Cladding())
        self.rect3 = geometry.Rectangle(10., 0.01, Barrier())
        self.rect4 = geometry.Rectangle(10., 0.01, Well())
        self.rect4.role = 'QW'
        self.rect5 = geometry.Rectangle(10., 0.01, Barrier())
        self.rect6 = geometry.Rectangle(10., 0.01, Cladding())
        self.rect7 = geometry.Rectangle(10., 0.02, Cap())
        stackSub = geometry.Stack2D()
        stackAct = geometry.Stack2D()
        stackAll = geometry.Stack2D()
        stackSub.append(self.rect1)
        stackAct.append(self.rect2)
        stackAct.append(self.rect3)
        stackAct.append(self.rect4)
        stackAct.append(self.rect5)
        stackAct.append(self.rect6)
        stackAct.role='active'
        stackAll.append(stackSub)
        stackAll.append(stackAct)
        stackAll.append(self.rect7)

        space = geometry.Cartesian2D(stackAct, left="mirror")
        self.solver.geometry = space

        self.solver.lifetime = 0.10
        self.solver.roughness = 0.05
        self.solver.matrix_elem = 10.
        self.solver.matrix_elem_sc_fact = 1.
        self.solver.cond_qw_shift = 0.000
        self.solver.vale_qw_shift = 0.000
        self.solver.strains = False
        self.solver.build_struct_once = False
        self.solver.Tref = 300.

    def testComputationsSpectrum(self):

        self.solver.inTemperature = 300.0
        self.solver.inCarriersConcentration = 4e18

        self.solver.invalidate()

        wavelengths = linspace(1500, 2500, 101)
        print wavelengths
        peak = 0.
        actlevel = 0.025
        spectrum = self.solver.spectrum(5, actlevel)
        gains = spectrum(wavelengths)
        peak = max(zip(wavelengths,gains), key=lambda wg: wg[1])
        print "Gain peak:", peak[0]
        print gains
        self.assertAlmostEqual( peak[0], 2180, 1 )

    """def testComputationsGainVal(self):

        self.solver.inTemperature = 300.0
        self.solver.inCarriersConcentration = 4e18

        e = 0.001
        axis0 = plask.mesh.Regular(0.+e, 10.-e, 3)
        axis1 = plask.mesh.Regular(0.+e, 0.050-e, 3)
        msh = mesh.Rectangular2D(axis0, axis1)

        print "Mesh:"
        for i in range(9):
            print msh[i]

        self.solver.inTemperature = 300.

        print "Outgain:"
        outgain = self.solver.outGain(msh,2000.)
        for i in range(9):
            print "[",i,"]:",outgain[i],"cm-1"

        self.assertAlmostEqual( outgain[1], 2527.548, 2 )
        self.assertAlmostEqual( outgain[4], 2527.548, 2 )
        self.assertAlmostEqual( outgain[7], 2527.548, 2 )"""

