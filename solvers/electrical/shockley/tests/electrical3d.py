#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from electrical.shockley import Shockley3D

eps0 = 8.854187817e-6 # pF/µm

config.axes = 'xyz'

@material.simple()
class Conductor(material.Material):
    def cond(self, T):
        return (1e+9, 1e+9)
    def eps(self, T):
        return 1.


class Shockley3D_Test(unittest.TestCase):

    def setUp(self):
        cond = Conductor()
        contact = geometry.Cuboid(700., 700., 1., cond)
        #rect = geometry.Cylinder(1000., 300., cond)
        #junc = geometry.Cylinder(1000., 0.02, 'GaAs')
        #self.S = pi * 1e6
        rect = geometry.Cuboid(1000., 1000., 300., cond)
        junc = geometry.Cuboid(1000., 1000., 0.02, 'GaAs')
        self.S = 1e6
        junc.role = 'active'
        stack = geometry.Stack3D(xcenter=0,  ycenter=0)
        top = stack.prepend(contact)
        stack.prepend(rect)
        stack.prepend(junc)
        stack.prepend(rect)
        bottom = stack.prepend(contact)
        space = geometry.Cartesian3D(stack)
        self.solver = Shockley3D("electrical3d")
        self.solver.geometry = space
        self.solver.mesh = mesh.Rectangular3D.DivideGenerator(prediv=(3,3,2), gradual=False)
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.maxerr = 1e-3
        self.solver.voltage_boundary.append(self.solver.mesh.TopOf(contact,top), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.BottomOf(contact, bottom), 1.)
        self.solver.algorithm = 'gauss'
        
    def testComputations(self):
        self.solver.compute(1000)
        correct_current = 1e-9 * self.S * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual( self.solver.get_total_current(), correct_current, 3 )
        capacitance = eps0 * material.GaAs().eps() * self.S / 0.02 # pF
        self.assertAlmostEqual( self.solver.get_capacitance(), capacitance, 2 )
        heat = correct_current * 1.
        self.assertAlmostEqual( self.solver.get_total_heat(), heat, 3 )

    def testConductivity(self):
        msh = self.solver.mesh.get_midpoints()
        geo = self.solver.geometry
        conds = [geo.get_material(point).cond(300.) if not geo.has_role('active', point) else (0.,5.) for point in msh]
        self.assertListEqual( list(self.solver.outConductivity(msh)), conds )

    if __name__ == '__main__':
        def testGeometry(self):
            for plane in ('xy', 'xz', 'yz'):
                fig = figure()
                plot_geometry(self.solver.geometry, plane=plane, margin=0.1, color='k')
                plot_mesh(self.solver.mesh, plane=plane, color='c')
                plot_boundary(self.solver.voltage_boundary, self.solver.mesh, self.solver.geometry, plane=plane,
                              #cmap='YlGnBu_r')
                              color='m')
                fig.canvas.set_window_title(plane)


if __name__ == '__main__':
    test = unittest.main(exit=False)
    show()
    sys.exit(not test.result.wasSuccessful())
