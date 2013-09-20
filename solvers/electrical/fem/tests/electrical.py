#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from electrical.fem import Shockley2D, ShockleyCyl

@material.simple
class Conductor(material.Material):
    def cond(self, T):
        return 1e+9


class Shockley2D_Test(unittest.TestCase):

    def setUp(self):
        rect = geometry.Rectangle(1000., 300., Conductor())
        junc = geometry.Rectangle(1000., 0.02, None)
        junc.role = 'active'
        stack = geometry.Stack2D()
        stack.append(rect)
        stack.append(junc)
        stack.append(rect)
        space = geometry.Cartesian2D(stack, length=1000.)
        self.solver = Shockley2D("electrical2d")
        self.solver.geometry = space
        generator = mesh.Rectilinear2D.DivideGenerator()
        generator.prediv = 2,1
        self.solver.mesh = generator
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.maxerr = 1e-9
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)

    def testComputations(self):
        self.solver.compute()
        correct_current = 1e-3 * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual( self.solver.get_total_current(), correct_current, 3 )



class ShockleyCyl_Test(unittest.TestCase):

    def setUp(self):
        rect = geometry.Rectangle(1000., 300., Conductor())
        junc = geometry.Rectangle(1000., 0.02, None)
        junc.role = 'active'
        stack = geometry.Stack2D()
        stack.append(rect)
        stack.append(junc)
        stack.append(rect)
        space = geometry.Cylindrical2D(stack)
        self.solver = ShockleyCyl("electricalcyl")
        self.solver.geometry = space
        generator = mesh.Rectilinear2D.DivideGenerator()
        generator.prediv = 4,1
        self.solver.mesh = generator
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.maxerr = 1e-9
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)

    def testComputations(self):
        self.solver.compute(1000)
        correct_current = 1e-3 * pi * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual( self.solver.get_total_current(), correct_current, 3 )

