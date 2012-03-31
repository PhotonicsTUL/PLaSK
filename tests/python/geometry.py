#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import sys
if sys.version < "2.7":
    unittest.TestCase.assertIsNone = lambda self, value: self.assertTrue(item is None)
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)

import plask, plask.materials, plask.geometry


class SimpleGeometry(unittest.TestCase):

    def setUp(self):
        @plask.materials.simple
        class Dumb(plask.materials.Material): pass

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
                <!--<stack2d name="stack">-->
                    <child><rectangle name="block" x="4" y="2" material="Dumb" /></child>
                    <ref name="block" />
                </stack2d>
            </geometry>
        ''')
        self.assertEqual( type(geometry.element("block")), plask.geometry.Block2D )
        print list(geometry.element("stack").getLeafsBoundigBoxes())
        self.assertEqual( list(geometry.element("stack").getLeafsBoundigBoxes()),
            [plask.geometry.Box2D(-2,0,2,2), plask.geometry.Box2D(-2,2,2,4), plask.geometry.Box2D(-2,4,2,6), plask.geometry.Box2D(-2,6,2,8)])
        if sys.version >= "2.7":
            with self.assertRaises(KeyError): geometry.element("nonexistent")

class GeometryObjects(unittest.TestCase):

    def setUp(self):
        @plask.materials.simple
        class Mat(plask.materials.Material): pass

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

    def testBackround(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block53)

        background = plask.geometry.Background2D(child=stack)
        self.assertEqual( background.getMaterial(-3.,1.), self.mat )
        self.assertEqual( background.getMaterial(3.,1.), self.mat )
        self.assertEqual( background.getMaterial(0.,-1.), self.mat )
        self.assertEqual( background.getMaterial(0.,4.), self.mat )

        background = plask.geometry.Background2D(child=stack, along='z')
        self.assertEqual( background.getMaterial(-3.,1.), None )
        self.assertEqual( background.getMaterial(3.,1.), None )
        self.assertEqual( background.getMaterial(0.,-1.), self.mat )
        self.assertEqual( background.getMaterial(0.,4.), self.mat )

        geometry = plask.geometry.Geometry()
        geometry.read('''
            <geometry axis="xy">
                <background2d name="back" along="y">
                    <stack2d name="stack">
                        <child><rectangle name="rect" x="4" y="2" material="GaN"/></child>
                    </stack2d>
                </background2d>
            </geometry>
        ''')
        background = geometry.element("back")
        self.assertEqual( background.getMaterial(-3.,1.), None )
        self.assertEqual( background.getMaterial(3.,1.), None )
        self.assertEqual( background.getMaterial(0.,-1.).name, "GaN" )
        self.assertEqual( background.getMaterial(0.,3.).name, "GaN" )


    def testMultiStack(self):
        multistack = plask.geometry.MultiStack2D(5, 10.0)
        multistack.append(self.block53)
        multistack.append(self.block53)
        self.assertIn( self.block53, multistack )
        # 5 * 2 childs = 10 elements, each have size 5x3, should be in [0, 10] - [5, 40]
        self.assertEqual( multistack.boundingBox, plask.geometry.Box2D(-2.5, 10.0, 2.5, 40.0) )
        self.assertEqual( multistack.getMaterial(1.0, 39.0), self.mat )
        self.assertIsNone( multistack.getMaterial(4.0, 41.0) )
        #self.assertEqual( multistack[0].child, self.block53 )
        #self.assertEqual( multistack[0].translation, plask.vec(0, 10.) )
        #self.assertEqual( multistack.repeatedItem(9).child, self.block53 )
        #self.assertEqual( multistack.repeatedItem(9).translation, plask.vec(0, 37.) )
        #if sys.version >= 2.7:
            #with self.assertRaises(IndexError): multistack[9]

    def testAligners(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block53, "C")
        stack.append(self.block53, "L")
        stack.append(self.block53, "R")
        self.assertEqual( stack.getMaterial(-1.0, 1.0), self.mat )
        self.assertEqual( stack.getMaterial(2.6, 1.0), None )
        self.assertEqual( stack.getMaterial(4.9, 4.0), self.mat )
        self.assertEqual( stack.getMaterial(-0.1, 4.0), None )
        self.assertEqual( stack.getMaterial(5.1, 4.0), None )
        self.assertEqual( stack.getMaterial(-4.9, 7.0), self.mat )
        self.assertEqual( stack.getMaterial(-5.1, 7.0), None )
        self.assertEqual( stack.getMaterial(0.1, 7.0), None )
        self.assertEqual( list(stack.getLeafsBoundigBoxes()), [plask.geometry.Box2D(-2.5,0,2.5,3), plask.geometry.Box2D(0.0,3,5.0,6), plask.geometry.Box2D(-5.0,6,0.0,9)])


class GeometryPath(unittest.TestCase):

    def setUp(self):
        self.stack1 = plask.geometry.Stack2D()
        self.stack2 = plask.geometry.Stack2D()
        self.element = plask.geometry.Rectangle(1,2, plask.materials.GaN())
        self.stack1.append(self.stack2)
        self.stack2.append(self.element)

    def testPath(self):
        p = plask.geometry.Path([self.stack1, self.stack2])
        p += self.element
