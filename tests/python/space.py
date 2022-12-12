#!/usr/bin/env plask
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
import sys

import plask, plask.geometry
import plasktest


class Geometrys(unittest.TestCase):

    def setUp(self):
        self.axes_backup = plask.config.axes
        plask.config.axes = "xy"

    def tearDown(self):
        plask.config.axes = self.axes_backup


    def testBorders(self):
        r = plask.geometry.Rectangle(2.,1., "Al(0.2)GaN")
        s = plask.geometry.Cartesian2D(r, x_lo="mirror" , right="AlN", top="GaN")
        print(s.bbox)
        self.assertEqual(dict(s.edges.items()), {'left': "mirror", 'right': "AlN", 'top': "GaN", 'bottom': None})
        self.assertEqual(str(s.get_material(-1.5, 0.5)), "Al(0.2)GaN")
        self.assertEqual(str(s.get_material(3., 0.5)), "AlN")
        self.assertEqual(str(s.get_material(-3., 0.5)), "AlN")
        self.assertEqual(str(s.get_material(0., 2.)), "GaN")

        with self.assertRaises(RuntimeError): plask.geometry.Cartesian2D(r, right="mirror").get_material(3., 0.5)

    #def testSubspace(self):
        #stack = plask.geometry.Stack2D()
        #r1 = plask.geometry.Rectangle(2.,2., "GaN")
        #r2 = plask.geometry.Rectangle(2.,1., "Al(0.2)GaN")
        #stack.append(r1, "l")
        #h2 = stack.append(r2, "l")
        #space = plask.geometry.Cartesian2D(stack)
        #subspace = space.getSubspace(r2)
        #v1 = space.getLeafsPositions(h2)
        #v2 = subspace.getLeafsPositions(h2)
        #self.assertEqual(space.getLeafsPositions(h2)[0], subspace.getLeafsPositions(h2)[0])

    def testSolver(self):
        solver = plasktest.SpaceTest()
        r = plask.geometry.Rectangle(2.,1., "Al(0.2)GaN")
        s = plask.geometry.Cartesian2D(r)
        solver.geometry = s
        self.assertEqual(solver.geometry, s)


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
