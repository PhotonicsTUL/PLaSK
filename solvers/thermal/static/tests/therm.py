#!/usr/bin/env python3
# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from thermal.static import Static2D

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
        for p in self.solver.mesh.elements.mesh:
            print(self.solver.geometry.get_material(p))
        conds = self.solver.outThermalConductivity(msh)
        self.assertAlmostEqual(conds[0][0], 2.000000, 6)
        self.assertAlmostEqual(conds[1][0], 2.000000, 6)
