#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from electrical.fem3d import Shockley3D

eps0 = 8.854187817e-6 # pF/µm

@material.simple
class Conductor(material.Material):
    def cond(self, T):
        return (1e+9, 1e+9)
    def eps(self, T):
        return 1.


class Shockley3D_Test(unittest.TestCase):

    def setUp(self):
        rect = geometry.Cuboid(1000., 1000., 300., Conductor())
        junc = geometry.Cuboid(1000., 1000., 0.02, 'air')
        junc.role = 'active'
        stack = geometry.Stack3D()
        stack.append(rect)
        stack.append(junc)
        stack.append(rect)
        space = geometry.Cartesian3D(stack)
        self.solver = Shockley3D("electrical3d")
        self.solver.geometry = space
        self.solver.mesh = mesh.Rectilinear3D.DivideGenerator(prediv=(2,2,2), gradual=False)
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.maxerr = 1e-5
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)
        self.solver.algorithm = 'gauss'

    def testComputations(self):
        self.solver.compute(1000)
        correct_current = 1e-3 * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual( self.solver.get_total_current(), correct_current, 3 )
        capacitance = eps0 * 1000.**2 / 0.02 # pF
        self.assertAlmostEqual( self.solver.get_capacitance(), capacitance, 4 )

    def testConductivity(self):
        msh = self.solver.mesh.get_midpoints()
        geo = self.solver.geometry
        conds = [geo.get_material(point).cond(300.) if not geo.has_role('active', point) else (0.,5.) for point in msh]
        self.assertListEqual( list(self.solver.outConductivity(msh)), conds )
