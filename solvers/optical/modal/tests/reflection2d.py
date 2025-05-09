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
from optical.modal import Fourier2D

plask.config.axes = 'xy'

@material.simple()
class Core(material.Material):
    Nr = 3.5


class Reflection2D_Test(unittest.TestCase):

    def setUp(self):
        self.solver = Fourier2D("fourier2d")
        rect = geometry.Rectangle(0.5, 0.25, Core())
        air = geometry.Rectangle(1.5, 0.50, "air")
        stack = geometry.Stack2D()
        stack.append(rect)
        stack.append(rect)
        shelf = geometry.Shelf()
        shelf.append(stack)
        shelf.append(air)
        space = geometry.Cartesian2D(shelf, left="mirror")
        self.solver.geometry = space
        self.solver.set_interface(0.25)
        self.solver.symmetry = 'Ex'
        self.solver.size = 32
        self.solver.smooth = 1e-5

    def testComputations(self):
        self.solver.wavelength = 1550.
        #X = linspace(3.00, 3.50, 200)
        #plot(X, [abs(self.solver.determinant(neff=x)) for x in X])
        #show()
        nm = self.solver.find_mode(neff=3.197)
        if __name__ == '__main__':
            figure()
            plot_field(self.solver.outLightMagnitude(nm,
                mesh.Rectangular2D(mesh.Regular(-0.75, 0.75, 100), mesh.Regular(-0.25, 0.75, 100))))
            plot_geometry(self.solver.geometry, color='w', mirror=True)
            show()
        self.assertAlmostEqual(self.solver.modes[nm].neff, 3.197, 3)


if __name__ == '__main__':
    unittest.main()
