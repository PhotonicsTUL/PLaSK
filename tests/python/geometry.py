#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import sys
if sys.version < "2.7":
    unittest.TestCase.assertIsNone = lambda self, value: self.assertTrue(item is None)

import plask, plask.material, plask.geometry

class SimpleGeometry(unittest.TestCase):

    def setUp(self):
        @plask.material.new
        class Dumb(plask.material.Material): pass

    def testPrimitives(self):
        '''Test the properties of primitives'''
        r2a = plask.geometry.Box2D()
        r2a.lower = plask.vec(3., 2.)
        r2a.upper = plask.vec(1., 5.)
        r2a.fix()
        self.assertAlmostEqual( r2a.lower, plask.vec(1,2) )
        self.assertAlmostEqual( r2a.upper, plask.vec(3,5) )
        r2b = plask.geometry.Box2D(plask.vec(3., 2.), plask.vec(1., 5.))
        r2b.fix()
        self.assertAlmostEqual( r2b.lower, plask.vec(1,2) )
        self.assertAlmostEqual( r2b.upper, plask.vec(3,5) )
        r3a = plask.geometry.Box3D(3.,2.,1., 1.,5.,0.)
        r3b = plask.geometry.Box3D(plask.vec(1.,2.,0.), plask.vec(3.,5.,1.))
        self.assertEqual( r3a, r3b )

    def testManager(self):
        '''test geometry manager'''
        geometry = plask.geometry.Geometry()
        geometry.read('''
            <geometry axis="xy">
                <stack2d repeat="2" name="stack">
                    <child><rectangle name="block" x="4" y="2" material="Dumb" /></child>
                    <ref name="block" />
                </stack2d>
            </geometry>
        ''')
        self.assertEqual( type(geometry["block"]), plask.geometry.Rectangle )
        if sys.version >= "2.7":
            with self.assertRaises(KeyError): geometry["nonexistent"]

class GeometryObjects(unittest.TestCase):

    def setUp(self):
        @plask.material.new
        class Mat(plask.material.Material): pass

        self.mat = Mat()
        self.rectangle53 = plask.geometry.Rectangle(5,3, self.mat)

    def testRectangle(self):
        '''Test rectangle'''
        self.assertAlmostEqual( self.rectangle53.boundingBox.upper, plask.vec(5.0, 3.0) )
        self.assertAlmostEqual( self.rectangle53.boundingBox.lower, plask.vec(0.0, 0.0) )
        self.assertEqual( self.rectangle53.getMaterial(plask.vec(4.0, 2.0)), self.mat)
        self.assertIsNone( self.rectangle53.getMaterial(plask.vec(6.0, 2.0)));


    def testTranslation(self):
        '''Test translations of the objects'''
        translation = plask.geometry.Translation2D(self.rectangle53, plask.vec(10.0, 20.0))    # should be in [10, 20] - [15, 23]
        self.assertEqual( translation.boundingBox, plask.geometry.Box2D(plask.vec(10, 20), plask.vec(15, 23)) )
        self.assertEqual( translation.getMaterial(12.0, 22.0), self.mat);
        self.assertIsNone( translation.getMaterial(4.0, 22.0));

    def testMultiStack(self):
        multistack = plask.geometry.MultiStack2D(5, 10.0)
        multistack.append(self.rectangle53)
        multistack.append(self.rectangle53)
        # 5 * 2 childs = 10 elements, each have size 5x3, should be in [0, 10] - [5, 40]
        self.assertEqual(multistack.boundingBox, plask.geometry.Box2D(0.0, 10.0, 5.0, 40.0))
        self.assertEqual(multistack.getMaterial(4.0, 39.0), self.mat)
        self.assertIsNone( multistack.getMaterial(4.0, 41.0) )
        self.assertEqual( multistack[0][0], self.rectangle53 )
        self.assertAlmostEqual( multistack[0][1], plask.vec(0, 10.) )
        self.assertEqual( multistack[9][0], self.rectangle53 )
        self.assertAlmostEqual( multistack[9][1], plask.vec(0, 37.) )
