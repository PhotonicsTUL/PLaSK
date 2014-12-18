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
    def Eg(self, T=300., e=0., point='G'): return 0.8
    def VB(self, T=300., e=0., point='G', hole='H'): return 0.4
    def VB(self, T=300., e=0., point='G', hole='L'): return 0.4
    def CB(self, T=300., e=0., point='G'): return 1.2
    def Dso(self, T=300., e=0.): return 0.35
    def Me(self, T=300., e=0., point='G'): return 0.050
    def Mhh(self, T=300., e=0.): return 0.200
    def Mlh(self, T=300., e=0.): return 0.080  
    
@material.simple()
class Cladding(material.Material):
    def Eg(self, T=300., e=0., point='G'): return 1.2
    def VB(self, T=300., e=0., point='G', hole='H'): return 0.2
    def VB(self, T=300., e=0., point='G', hole='L'): return 0.2
    def CB(self, T=300., e=0., point='G'): return 1.4
    def Dso(self, T=300., e=0.): return 0.35
    def Me(self, T=300., e=0., point='G'): return 0.050
    def Mhh(self, T=300., e=0.): return 0.200
    def Mlh(self, T=300., e=0.): return 0.080  
    
@material.simple()
class Substrate(material.Material):
    def Eg(self, T=300., e=0., point='G'): return 1.5
    def VB(self, T=300., e=0., point='G', hole='H'): return 0.1
    def CB(self, T=300., e=0., point='G'): return 1.6
    def Dso(self, T=300., e=0.): return 0.35
    def Me(self, T=300., e=0., point='G'): return 0.050
    def Mhh(self, T=300., e=0.): return 0.200
    def Mlh(self, T=300., e=0.): return 0.080      

@material.simple()
class Cap(material.Material):
    def Eg(self, T=300., e=0., point='G'): return 1.5
    def VB(self, T=300., e=0., point='G', hole='H'): return 0.1
    def CB(self, T=300., e=0., point='G'): return 1.6
    def Dso(self, T=300., e=0.): return 0.35
    def Me(self, T=300., e=0., point='G'): return 0.050
    def Mhh(self, T=300., e=0.): return 0.200
    def Mlh(self, T=300., e=0.): return 0.080

class NewGainValues(unittest.TestCase):

    def setUp(self):
        self.solver = Ferminew2D("GAIN")
        plask.config.axes = 'xy'
        
        self.rect1 = geometry.Rectangle(10., 10., Substrate())
        self.rect1.role = 'substrate'
        self.rect2 = geometry.Rectangle(10., 50e-3, Cladding())
        self.rect3 = geometry.Rectangle(10., 8e-3, Barrier())
        self.rect4 = geometry.Rectangle(10., 11e-3, Well())
        self.rect4.role = 'QW'
        self.rect5 = geometry.Rectangle(10., 8e-3, Barrier())
        self.rect6 = geometry.Rectangle(10., 50e-3, Cladding())
        self.rect7 = geometry.Rectangle(10., 20e-3, Cap())
        stackSub = geometry.Stack2D()
        stackAct = geometry.Stack2D()
        stackAll = geometry.Stack2D()
        stackSub.prepend(self.rect1)
        stackAct.prepend(self.rect2)
        stackAct.prepend(self.rect3)
        stackAct.prepend(self.rect4)
        stackAct.prepend(self.rect5)
        stackAct.prepend(self.rect6)
        stackAct.role='active'
        stackAll.prepend(stackSub)
        stackAll.prepend(stackAct)
        stackAll.prepend(self.rect7)

        space = geometry.Cartesian2D(stackAct, left="mirror")
        self.solver.geometry = space
        
    def testComputationsSpectrum(self):
        
        self.solver.lifetime = 0.10
        self.solver.roughness = 0.05
        self.solver.matrix_elem = 10.
        self.solver.matrix_elem_sc_fact = 1.
        self.solver.cond_qw_shift = 0.000
        self.solver.vale_qw_shift = 0.000
        self.solver.strains = False
        self.solver.build_struct_once = False
        self.solver.Tref = 300.        

        self.solver.inTemperature = 300.0
        self.solver.inCarriersConcentration = 2e18

        wavelengths = linspace(1000, 4000, 3001)
        peak = 0.
        actlevel = 0.0635
        spectrum = self.solver.spectrum(0, actlevel)
        gains = spectrum(wavelengths)
        peak = max(zip(wavelengths,gains), key=lambda (w,g): g)
        print "Gain peak:", peak[0]

        #axis0 = plask.mesh.Regular(0., 10., 100)
        #axis1 = plask.mesh.Regular(0., 0.127, 128)
        #msh = mesh.Rectangular2D(axis0, axis1)


        #mesh = self.solver.mesh // earlier
        #outgain = self.solver.outGain(msh,10.)
        #print outgain

        #self.assertAlmostEqual( 2211.000, 2.001, 1 )
        self.assertAlmostEqual( peak[0], 2211, 1 )
