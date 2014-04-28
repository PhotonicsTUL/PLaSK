#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.slab import FourierReflection2D

@material.simple
class Glass(material.Material):

    def Nr(*args): return 1.3

@material.simple
class Cladding(material.Material):

    def Nr(*args): return 1.28



class Reflection2D_Test(unittest.TestCase):

    def setUp(self):
        self.solver = FourierReflection2D("fourier2d")
        rect = geometry.Rectangle(0.75, 0.25, Glass())
        stack = geometry.Stack2D()
        pth = stack.append(rect)
        stack.append(rect)
        space = geometry.Cartesian2D(stack, left="mirror")
        self.solver.geometry = space
        self.solver.set_interface(rect, pth)

    def testComputations(self):
        self.solver.wavelength = 1000.
        show()
        self.assertAlmostEqual( self.solver.modes[self.solver.find_mode(1.148)].neff, 1.147, 3 )
