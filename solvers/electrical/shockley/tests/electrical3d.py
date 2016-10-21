#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from electrical.shockley import Shockley3D

eps0 = 8.854187817e-6 # pF/µm

@material.simple()
class Conductor(material.Material):
    def cond(self, T):
        return (1e+9, 1e+9)
    def eps(self, T):
        return 1.


class Shockley3D_Test(unittest.TestCase):

    def setUp(self):
        rect = geometry.Cuboid(2000., 1000., 300., Conductor())
        junc = geometry.Cuboid(2000., 1000., 0.02, 'GaAs')
        junc.role = 'active'
        stack = geometry.Stack3D()
        stack.append(rect)
        stack.append(junc)
        stack.append(rect)
        space = geometry.Cartesian3D(stack)
        self.solver = Shockley3D("electrical3d")
        self.solver.geometry = space
        self.solver.mesh = mesh.Rectangular3D.DivideGenerator(prediv=(3,2,2), gradual=False)
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.maxerr = 1e-5
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)
        self.solver.algorithm = 'gauss'

    def testComputations(self):
        self.solver.compute(1000)
        correct_current = 2e-3 * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual( self.solver.get_total_current(), correct_current, 3 )
        capacitance = eps0 * material.GaAs().eps() * 1000. * 2000. / 0.02 # pF
        self.assertAlmostEqual( self.solver.get_capacitance(), capacitance, 2 )
        heat = correct_current * 1.
        self.assertAlmostEqual( self.solver.get_total_heat(), heat, 3 )

    def testConductivity(self):
        msh = self.solver.mesh.get_midpoints()
        geo = self.solver.geometry
        conds = [geo.get_material(point).cond(300.) if not geo.has_role('active', point) else (0.,5.) for point in msh]
        self.assertListEqual( list(self.solver.outConductivity(msh)), conds )
