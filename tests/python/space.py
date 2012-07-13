#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest


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
        self.assertEqual( s.borders, {'left': "mirror", 'right': "AlN", 'top': "GaN", 'bottom': None} )
        self.assertEqual( str(s.getMaterial(-1.5, 0.5)), "Al(0.2)GaN" )
        self.assertEqual( str(s.getMaterial(3., 0.5)), "AlN" )
        self.assertEqual( str(s.getMaterial(-3., 0.5)), "AlN" )
        self.assertEqual( str(s.getMaterial(0., 2.)), "GaN" )

        with self.assertRaises(RuntimeError): plask.geometry.Cartesian2D(r, right="mirror").getMaterial(3., 0.5)

    def testSubspace(self):
        stack = plask.geometry.Stack2D()
        r1 = plask.geometry.Rectangle(2.,2., "GaN")
        r2 = plask.geometry.Rectangle(2.,1., "Al(0.2)GaN")
        stack.append(r1, "l")
        h2 = stack.append(r2, "l")
        space = plask.geometry.Cartesian2D(stack)
        subspace = space.getSubspace(r2)
        v1 = space.getLeafsPositions(h2)
        v2 = subspace.getLeafsPositions(h2)
        self.assertEqual( space.getLeafsPositions(h2)[0], subspace.getLeafsPositions(h2)[0] )

    def testModule(self):
        module = plasktest.SpaceTest()
        r = plask.geometry.Rectangle(2.,1., "Al(0.2)GaN")
        s = plask.geometry.Cartesian2D(r)
        module.geometry = s
        self.assertEqual( module.geometry, s )
        with self.assertRaises(AttributeError):
            print module.mesh
