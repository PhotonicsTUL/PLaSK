#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import sys
if sys.version < "2.7":
    unittest.TestCase.assertIsNone = lambda self, value: self.assertTrue(item is None)
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)

import plask, plask.material, plask.geometry

class SimpleGeometry(unittest.TestCase):

    def setUp(self):
        @plask.material.simple
        class Dumb(plask.material.Material): pass

    def testPrimitives(self):
        '''Test the properties of primitives'''
        r2a = plask.geometry.Box2D()
        r2a.lower = plask.vec(3., 2.)
        r2a.upper = plask.vec(1., 5.)
        r2a.fix()
        self.assertEqual( r2a.lower, plask.vec(1,2) )
        self.assertEqual( r2a.upper, plask.vec(3,5) )
        r2b = plask.geometry.Box2D(plask.vec(3., 2.), plask.vec(1., 5.))
        r2b.fix()
        self.assertEqual( r2b.lower, plask.vec(1,2) )
        self.assertEqual( r2b.upper, plask.vec(3,5) )
        r3a = plask.geometry.Box3D(3.,2.,1., 1.,5.,0.)
        r3b = plask.geometry.Box3D(plask.vec(1.,2.,0.), plask.vec(3.,5.,1.))
        self.assertEqual( r3a, r3b )

    def testManager(self):
        '''test geometry manager'''
        geometry = plask.geometry.Geometry()
        geometry.read('''
            <geometry axis="xy">
                <stack2d repeat="2" name="stack">
                    <child><block2d name="block" x="4" y="2" material="Dumb" /></child>
                    <ref name="block" />
                </stack2d>
            </geometry>
        ''')
        self.assertEqual( type(geometry.element("block")), plask.geometry.Block2D )
#        for l,r in zip( geometry.element("stack").leafsBoundigBoxes(), [plask.geometry.Box2D(0,0,4,2), plask.geometry.Box2D(0,2,4,4)]):
#            self.assertEqual(l, r)
        if sys.version >= "2.7":
            with self.assertRaises(KeyError): geometry.element("nonexistent")

class GeometryObjects(unittest.TestCase):

    def setUp(self):
        @plask.material.simple
        class Mat(plask.material.Material): pass

        self.mat = Mat()
        self.block53 = plask.geometry.Block2D(5,3, self.mat)

    def testRectangle(self):
        '''Test rectangle'''
        self.assertEqual( self.block53.boundingBox.upper, plask.vec(5.0, 3.0) )
        self.assertEqual( self.block53.boundingBox.lower, plask.vec(0.0, 0.0) )
        self.assertEqual( self.block53.getMaterial(plask.vec(4.0, 2.0)), self.mat)
        self.assertIsNone( self.block53.getMaterial(plask.vec(6.0, 2.0)));


    def testTranslation(self):
        '''Test translations of the objects'''
        translation = plask.geometry.Translation2D(self.block53, plask.vec(10.0, 20.0))    # should be in [10, 20] - [15, 23]
        self.assertEqual( translation.boundingBox, plask.geometry.Box2D(plask.vec(10, 20), plask.vec(15, 23)) )
        self.assertEqual( translation.getMaterial(12.0, 22.0), self.mat);
        self.assertIsNone( translation.getMaterial(4.0, 22.0));

    def testMultiStack(self):
        multistack = plask.geometry.MultiStack2D(5, 10.0)
        multistack.append(self.block53)
        multistack.append(self.block53)
        self.assertIn( self.block53, multistack )
        # 5 * 2 childs = 10 elements, each have size 5x3, should be in [0, 10] - [5, 40]
        self.assertEqual(multistack.boundingBox, plask.geometry.Box2D(0.0, 10.0, 5.0, 40.0))
        self.assertEqual(multistack.getMaterial(4.0, 39.0), self.mat)
        self.assertIsNone( multistack.getMaterial(4.0, 41.0) )
        #self.assertEqual( multistack[0].child, self.block53 )
        #self.assertEqual( multistack[0].translation, plask.vec(0, 10.) )
        #self.assertEqual( multistack.repeatedItem(9).child, self.block53 )
        #self.assertEqual( multistack.repeatedItem(9).translation, plask.vec(0, 37.) )
        #if sys.version >= 2.7:
            #with self.assertRaises(IndexError): multistack[9]
