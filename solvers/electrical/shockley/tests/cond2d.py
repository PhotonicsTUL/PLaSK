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
from electrical.shockley import Shockley2D, ShockleyCyl, ActiveCond2D

eps0 = 8.854187817e-6 # pF/µm

@material.simple()
class Conductor(material.Material):
    def cond(self, T):
        return (1e+12, 1e+12)
    def eps(self, T):
        return 1.


class Cond2D_Test(unittest.TestCase):

    def setUp(self):
        rect = geometry.Rectangle(1000., 300., Conductor())
        junc = geometry.Rectangle(1000., 0.2, 'GaAs')
        junc.role = 'active'
        stack = geometry.Stack2D()
        stack.append(rect)
        stack.append(junc)
        stack.append(rect)
        space = geometry.Cartesian2D(stack, length=1000.)
        self.solver = ActiveCond2D("electrical2d")
        self.solver.geometry = space
        generator = mesh.Rectangular2D.DivideGenerator()
        generator.prediv = 2,1
        self.solver.mesh = generator
        self.solver.cond = lambda U, j, T: 0.05 + abs(j)
        self.solver.maxerr = 1e-5
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)

    def testComputations(self):
        self.solver.compute()
        correct_current = 500.
        self.assertAlmostEqual(self.solver.get_total_current(), correct_current, 3)


if __name__ == '__main__':
    test = unittest.main(exit=False)
    show()
    sys.exit(not test.result.wasSuccessful())
