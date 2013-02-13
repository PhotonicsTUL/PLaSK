#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

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



class GeometryObjects(unittest.TestCase):

    def setUp(self):
        @plask.material.simple
        class Mat(plask.material.Material):
            pass

        self.mat = Mat()
        self.block53 = plask.geometry.Block2D(5,3, self.mat)

    def testRectangle(self):
        '''Test rectangle'''
        self.assertEqual( self.block53.bbox.upper, plask.vec(5.0, 3.0) )
        self.assertEqual( self.block53.bbox.lower, plask.vec(0.0, 0.0) )
        self.assertIs( self.block53.get_material(plask.vec(4.0, 2.0)), self.mat)
        self.assertIsNone( self.block53.get_material(plask.vec(6.0, 2.0)));



class Transforms(unittest.TestCase):

    def setUp(self):
        self.mat = plask.material.GaN()
        self.block53 = plask.geometry.Block2D(5,3, self.mat)

    def testTranslation(self):
        '''Test translations of the objects'''
        translation = plask.geometry.Translation2D(self.block53, plask.vec(10.0, 20.0))    # should be in [10, 20] - [15, 23]
        self.assertEqual( translation.bbox, plask.geometry.Box2D(plask.vec(10, 20), plask.vec(15, 23)) )
        self.assertEqual( translation.get_material(12.0, 22.0), self.mat);
        self.assertIsNone( translation.get_material(4.0, 22.0));



class GeometryPath(unittest.TestCase):

    def setUp(self):
        self.stack1 = plask.geometry.Stack2D()
        self.stack2 = plask.geometry.Stack2D()
        self.object1 = plask.geometry.Rectangle(1,2, plask.material.GaN())
        self.object2 = plask.geometry.Rectangle(1,2, plask.material.GaN())
        self.stack1.append(self.stack2)
        self.stack2.append(self.object1)
        self.stack2.append(self.object2)

    def testPath(self):
        p = plask.geometry.Path([self.stack1, self.stack2])
        p += self.object1

    def testIncludes(self):
        self.assertTrue( self.stack1.object_includes(self.object1, 0.5, 1.0) )
        self.assertFalse( self.stack1.object_includes(self.object1, 0.5, 3.0) )


class Containers(unittest.TestCase):

    def setUp(self):
        self.gan = plask.material.GaN()
        self.aln = plask.material.AlN()
        self.block1 = plask.geometry.Block2D(5,3, self.gan)
        self.block2 = plask.geometry.Block2D(5,3, self.aln)
        self.cube1 = plask.geometry.Block3D(4,4,2, self.gan)
        self.cube2 = plask.geometry.Block3D(4,4,2, self.aln)

    def testAligners(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block1, ycenter=0)
        stack.append(self.block1, left=0)
        stack.append(self.block1, right=0)
        self.assertEqual( stack.get_material(-1.0, 1.0), self.gan )
        self.assertEqual( stack.get_material(2.6, 1.0), None )
        self.assertEqual( stack.get_material(4.9, 4.0), self.gan )
        self.assertEqual( stack.get_material(-0.1, 4.0), None )
        self.assertEqual( stack.get_material(5.1, 4.0), None )
        self.assertEqual( stack.get_material(-4.9, 7.0), self.gan )
        self.assertEqual( stack.get_material(-5.1, 7.0), None )
        self.assertEqual( stack.get_material(0.1, 7.0), None )
        self.assertEqual( list(stack.get_leafs_bboxes()), [plask.geometry.Box2D(-2.5,0,2.5,3), plask.geometry.Box2D(0.0,3,5.0,6), plask.geometry.Box2D(-5.0,6,0.0,9)])

    def testMultiStack(self):
        multistack = plask.geometry.MultiStack2D(5, 10.0)
        hint1 = multistack.append(self.block1)
        hint2 = multistack.append(self.block2)
        self.assertEqual( multistack.get_material(0.,10.), self.block1.get_material(0.,0.) )

        self.assertIn( self.block1, multistack )
        # 5 * 2 childs = 10 objects, each have size 5x3, should be in [0, 10] - [5, 40]
        self.assertEqual( multistack.bbox, plask.geometry.Box2D(0., 10.0, 5., 40.0) )
        self.assertEqual( multistack.get_material(1.0, 39.0), self.aln )
        self.assertIsNone( multistack.get_material(4.0, 41.0) )
        self.assertEqual( multistack[0].child, self.block1 )
        self.assertEqual( multistack[0].translation, plask.vec(0., 10.) )
        self.assertEqual( multistack[9].child, self.block2 )
        self.assertEqual( multistack[9].translation, plask.vec(0., 37.) )
        hints1 = plask.geometry.PathHints()
        hints1 += hint1
        self.assertEqual( len(multistack[hints1]), 1 )
        self.assertEqual( multistack[hints1][0].child, self.block1 )
        self.assertEqual( multistack[hint2][0].child, self.block2 )

    def testRemoval(self):
        '''Test if removing objects from container works. In addition test prepending objects'''
        container = plask.geometry.TranslationContainer2D()
        h = container.append(self.block1, 0,0) # be removed by hint
        container.append(self.block2, 10,0)
        container.append(self.block1, 10,0) # to be removed by index
        container.append(self.block2, 5,0) # to be removed by object
        self.assertEqual( container[0].child, self.block1 )
        self.assertEqual( container.get_material(12,1), self.gan)
        del container[2]
        self.assertEqual( len(container), 3 )
        self.assertEqual( container.get_material(12,1), self.aln)
        del container[h]
        self.assertEqual( len(container), 2 )
        self.assertEqual( container.get_material(1,1), None )
        del container[container[-1]]
        self.assertEqual( len(container), 1 )
        self.assertEqual( container.get_material(6,1), None )


        stack = plask.geometry.Stack3D()
        stack.append(self.cube1, xcenter=0, ycenter=0) # to be removed by object
        stack.append(self.cube2, xcenter=0, ycenter=0) # to be removed by index
        h = stack.append(self.cube2, xcenter=0, ycenter=0) # to be removed by hint
        stack.append(self.cube1, xcenter=0, ycenter=0)
        self.assertEqual( len(stack), 4 )
        self.assertEqual( stack.bbox, plask.geometry.Box3D(-2,-2,0, 2,2,8) )
        self.assertEqual( stack.get_material(0,0,5), self.aln)
        del stack[1]
        self.assertEqual( len(stack), 3 )
        self.assertEqual( stack.bbox, plask.geometry.Box3D(-2,-2,0, 2,2,6) )
        self.assertEqual( stack.get_material(0,0,5), self.gan)
        self.assertEqual( stack.get_material(0,0,3), self.aln)
        del stack[h]
        self.assertEqual( len(stack), 2 )
        self.assertEqual( stack.bbox, plask.geometry.Box3D(-2,-2,0, 2,2,4) )
        self.assertEqual( stack.get_material(0,0,3), self.gan)
        del stack[stack[0]]
        self.assertEqual( len(stack), 1 )
        self.assertEqual( stack.bbox, plask.geometry.Box3D(-2,-2,0, 2,2,2) )

        self.assertEqual( stack.get_material(0,0,1), self.gan)
        stack.prepend(self.cube2, xcenter=0, ycenter=0)
        self.assertEqual( stack.bbox, plask.geometry.Box3D(-2,-2,0, 2,2,4) )
        self.assertEqual( stack.get_material(0,0,1), self.aln)
        self.assertEqual( stack.get_material(0,0,3), self.gan)

    def testStackZero(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block1)
        stack.append(self.block1)
        stack.append(self.block1)
        self.assertEqual( stack.bbox, geometry.Box2D(0,0, 5,9) )
        stack.set_zero_below(2)
        self.assertEqual( stack.bbox, geometry.Box2D(0,-6, 5,3) )


    def testRoles(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block1)
        stack.append(self.block2)
        self.block2.role = "something"
        self.assertIn( "something", stack.get_roles(2., 4.) )
        self.assertTrue( stack.has_role("something", 2., 4.) )


