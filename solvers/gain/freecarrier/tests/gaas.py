#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh

from gain import FreeCarrierCyl

plask.config.axes = 'rz'

@material.simple('semiconductor')
class Barrier(material.Material):
    
    @staticmethod
    def lattC():
        return 5.654
    
    @staticmethod
    def Dso():
        return 0.3548
    
    @staticmethod
    def Me():
        return 0.103
    
    @staticmethod
    def Mhh():
        return 0.6
    
    @staticmethod
    def Mlh():
        return 0.14
    
    @staticmethod
    def D():
        return 10

    @staticmethod
    def VB():
        return -0.75
    
    def A(self, T=300.):
        return 7e7 + (T-300) * 1.4e5
    
    def B(self, T=300.):
        return 1.1e-10 - (T-300) * -2.2e-13
    
    def C(self, T=300.):
        return 1.130976e-28 - (T-300) * 1.028743e-30 + (T-300)**2 * 8.694142e-32
    
    def Eg(self, T=300., e=0., p=0.):
        return 1.30 - 0.0003 * (T-300)

    def nr(self, wl, T=300., n=0.):
        return 3.3 + (wl-1310) * 5.00e-4 + (T-300) * 7e-4

    def absp(self, wl, T=300.):
        return 50. + (T-300) * 7e-2


@material.simple(Barrier())
class Well(material.Material):
    
    @staticmethod
    def Me():
        return 0.052
    
    @staticmethod
    def Mhh():
        return 0.477
    
    @staticmethod
    def Mlh():
        return 0.103
    
    @staticmethod
    def absp():
        return 1000.

    def VB(self, T=300., e=0., p=0., h=0.):
        return 0.14676 + 0.000033 * (T-300) - 0.75
    
    def Eg(self, T=300., e=0., p=0.):
        return 0.87 - 0.0002 * (T-300) - 100. * e
    
    def nr(self, wl, T=300., n=0.):
        return 3.6 + (wl-1310) * 5.00e-4 + (T-300) * 7e-4


class TestStructureGain(unittest.TestCase):
    
    def setUp(self):
        substrate = geometry.Rectangle(20., 1000., 'GaAs')
        substrate.role = 'substrate'
        well = geometry.Rectangle(20., 0.0060, 'Well')
        well .role = 'QW'
        barrier = geometry.Rectangle(20., 0.0067, 'Barrier')
        
        stack = geometry.Stack2D()
        stack.prepend(substrate)
        self.active = geometry.Stack2D()
        self.active.role = 'active'
        for i in range(4):
            self.active.prepend(barrier)
            self.active.prepend(well)
        self.active.prepend(barrier)
        stack.prepend(self.active)
        stack.prepend(substrate)
        geom = geometry.Cylindrical2D(stack)

        self.solver = FreeCarrierCyl("self.solver")
        self.solver.geometry = geom
        
        self.msh = mesh.Rectangular2D([0.], [1000.0100])
        
        self.solver.inCarriersConcentration = 3.5e18

    def plot_bands(self, el_color='c', hh_color='r', lh_color='y'):
        box = self.solver.geometry.get_object_bboxes(self.active)[0]
        zz = linspace(box.lower.z-0.002, box.upper.z+0.002, 1001)
        CC = [self.solver.geometry.get_material(0.,z).CB() for z in zz]
        VV = [self.solver.geometry.get_material(0.,z).VB() for z in zz]
        plot(1e3*zz, CC, color='c')
        plot(1e3*zz, VV, color='r')
        xlim(1e3*zz[0], 1e3*zz[-1])
        xlabel("$z$ [nm]")
        ylabel("Band Edges [eV]")
        window_title("Band Edges")
        tight_layout(0.5)

    def test_gain(self):
        self.assertAlmostEqual( self.solver.outGain(self.msh, 1275.)[0], 1254., 0 )


if __name__ == '__main__':
    test = unittest.main(exit=False)
    show()
    sys.exit(not test.result.wasSuccessful())
