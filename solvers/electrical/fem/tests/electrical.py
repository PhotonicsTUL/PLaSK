#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from electrical.fem import Beta2D, BetaCyl

@material.simple
class Conductor(material.Material):
    def cond(self, T):
        return 1e+9


class Beta2D_Test(unittest.TestCase):

    def setUp(self):
        rect = geometry.Rectangle(1000., 300., Conductor())
        junc = geometry.Rectangle(1000., 0.02, None)
        junc.role = 'active'
        stack = geometry.Stack2D()
        stack.append(rect)
        stack.append(junc)
        stack.append(rect)
        space = geometry.Cartesian2D(stack, length=1000.)
        self.solver = Beta2D("electrical2d")
        self.solver.geometry = space
        self.solver.mesh = mesh.Rectilinear2D.DivideGenerator()
        self.solver.mesh.prediv = 4,1
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.corrlim = 1e-15 # such insane accuracy on voltage is necessary
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)

    def testComputations(self):
        self.solver.compute()
        correct_current = 1e-3 * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual( self.solver.get_total_current(), correct_current, 4 )



class BetaCyl_Test(unittest.TestCase):

    def setUp(self):
        rect = geometry.Rectangle(1000., 300., Conductor())
        junc = geometry.Rectangle(1000., 0.02, None)
        junc.role = 'active'
        stack = geometry.Stack2D()
        stack.append(rect)
        stack.append(junc)
        stack.append(rect)
        space = geometry.Cylindrical2D(stack)
        self.solver = BetaCyl("electricalcyl")
        self.solver.geometry = space
        self.solver.mesh = mesh.Rectilinear2D.DivideGenerator()
        self.solver.mesh.prediv = 16,1
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.corrlim = 1e-18 # such insane accuracy on voltage is necessary
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)

    def testComputations(self):
        self.solver.compute()
        correct_current = 1e-3 * pi * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual( self.solver.get_total_current(), correct_current, 4 )

