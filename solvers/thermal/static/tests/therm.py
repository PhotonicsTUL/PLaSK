#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from thermal.fem import Static2D

plask.config.axes = 'xy'

@material.simple()
class Strange(material.Material):

    def thermk(self, T, h):
        return h


class Layer_Test(unittest.TestCase):

    def setUp(self):
        srect = geometry.Rectangle(2, 1.0, Strange())
        other = geometry.Rectangle(2, 0.2, 'GaAs')
        stack = geometry.Stack2D()
        stack.prepend(other)
        stack.prepend(srect)
        stack.prepend(srect)
        stack.prepend(other)
        geom = geometry.Cartesian2D(stack)
        self.solver = Static2D("therm2d")
        self.solver.geometry = geom
        msh = mesh.Rectangular2D.DivideGenerator()
        self.solver.mesh = msh

    def testConductivity(self):
        msh = mesh.Rectangular2D([1.], [0.7, 1.7])
        for p in self.solver.mesh.get_midpoints():
            print(self.solver.geometry.get_material(p))
        conds = self.solver.outThermalConductivity(msh)
        self.assertAlmostEqual( conds[0][0], 2.000000, 6 )
        self.assertAlmostEqual( conds[1][0], 2.000000, 6 )
