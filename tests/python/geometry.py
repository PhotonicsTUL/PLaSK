#!/usr/bin/env plask
# -*- coding: utf-8 -*-
import sys
import unittest

import plask, plask.material, plask.geometry


class SimpleGeometry(unittest.TestCase):

    def setUp(self):
        @plask.material.simple()
        class Dumb(plask.material.Material): pass

    def testPrimitives(self):
        '''Test the properties of primitives'''
        r2a = plask.geometry.Box2D(3.,2., 1.,5.)
        self.assertEqual( r2a.lower, plask.vec(1,2) )
        self.assertEqual( r2a.upper, plask.vec(3,5) )
        r2b = plask.geometry.Box2D(plask.vec(1., 2.), plask.vec(3., 5.))
        self.assertEqual( r2b.lower, plask.vec(1,2) )
        self.assertEqual( r2b.upper, plask.vec(3,5) )
        r3a = plask.geometry.Box3D(3.,2.,1., 1.,5.,0.)
        r3b = plask.geometry.Box3D(plask.vec(1.,2.,0.), plask.vec(3.,5.,1.))
        self.assertEqual( r3a, r3b )


class GeometryObjects(unittest.TestCase):

    def setUp(self):
        @plask.material.simple()
        class Mat(plask.material.Material):
            pass
        self.mat = Mat()
        self.block53 = plask.geometry.Block2D(5,3, self.mat)
        plask.config.axes = 'xy'

    def testRectangle(self):
        '''Test rectangle'''
        self.assertEqual( self.block53.bbox.upper, plask.vec(5., 3.) )
        self.assertEqual( self.block53.bbox.lower, plask.vec(0., 0.) )
        self.assertIs( self.block53.get_material(plask.vec(4., 2.)), self.mat)
        self.assertIsNone( self.block53.get_material(plask.vec(6., 2.)));
        self.block53.width *= 3
        self.assertEqual( self.block53.dims, plask.vec(15., 3.) )
        self.block53.dy *= 2
        self.assertEqual( self.block53.dims, plask.vec(15., 6.) )


class Transforms(unittest.TestCase):

    def setUp(self):
        self.mat = plask.material.GaN()
        self.block53 = plask.geometry.Block2D(5,3, self.mat)
        plask.config.axes = 'xy'

    def testTranslation(self):
        '''Test translations of the objects'''
        translation = plask.geometry.Translation2D(self.block53, plask.vec(10.0, 20.0))    # should be in [10, 20] - [15, 23]
        self.assertEqual( translation.bbox, plask.geometry.Box2D(plask.vec(10, 20), plask.vec(15, 23)) )
        self.assertEqual( translation.get_material(12.0, 22.0), self.mat);
        self.assertIsNone( translation.get_material(4.0, 22.0));


    def testFlipMirror(self):
        self.assertEqual(plask.geometry.Flip2D(0, self.block53).bbox, plask.geometry.Box2D((-5, 0), (0, 3)))
        self.assertEqual(plask.geometry.Flip2D('y', self.block53).bbox, plask.geometry.Box2D((0, -3), (5, 0)))
        self.assertEqual(plask.geometry.Mirror2D(0, self.block53).bbox, plask.geometry.Box2D((-5, 0), (5, 3)))
        self.assertEqual(plask.geometry.Mirror2D('y', self.block53).bbox, plask.geometry.Box2D((0, -3), (5, 3)))



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
        p = plask.geometry.Path(self.stack1)
        p += self.stack2
        p += self.object1

    def testIncludes(self):
        self.assertTrue( self.stack1.object_contains(self.object1, 0.5, 1.0) )
        self.assertFalse( self.stack1.object_contains(self.object1, 0.5, 3.0) )


class PathHints(unittest.TestCase):

    def testHints(self):
        rect = plask.geometry.Rectangle(1,1,'GaAs')
        stack = plask.geometry.Stack2D()
        hint = stack.append(rect)
        stack.append(rect)
        stack.append(rect)
        stack.append(rect)
        self.assertAlmostEqual( stack.get_leafs_positions(hint)[0][1], 0. )
        self.assertAlmostEqual( stack.get_leafs_positions((stack, 1))[0][1], 1. )
        self.assertAlmostEqual( stack.get_leafs_positions({stack: 2})[0][1], 2. )
        self.assertAlmostEqual( stack.get_leafs_positions({stack: (3,)})[0][1], 3. )
        del stack, hint
        self.assertEqual( sys.getrefcount(rect), 2 )


class Containers(unittest.TestCase):

    def setUp(self):
        self.gan = plask.material.GaN()
        self.aln = plask.material.AlN()
        self.block1 = plask.geometry.Block2D(5,3, self.gan)
        self.block2 = plask.geometry.Block2D(5,3, self.aln)
        self.block2.role = "something"
        self.cube1 = plask.geometry.Block3D(4,4,2, self.gan)
        self.cube2 = plask.geometry.Block3D(4,4,2, self.aln)
        plask.config.axes = 'yz'

    def testAligners(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block1, ycenter=0)
        stack.append(self.block1, left=0)
        third = stack.append(self.block1, right=0)
        self.assertEqual( stack.get_material(-1.0, 1.0), self.gan )
        self.assertEqual( stack.get_material(2.6, 1.0), None )
        self.assertEqual( stack.get_material(4.9, 4.0), self.gan )
        self.assertEqual( stack.get_material(-0.1, 4.0), None )
        self.assertEqual( stack.get_material(5.1, 4.0), None )
        self.assertEqual( stack.get_material(-4.9, 7.0), self.gan )
        self.assertEqual( stack.get_material(-5.1, 7.0), None )
        self.assertEqual( stack.get_material(0.1, 7.0), None )
        self.assertEqual( list(stack.get_leafs_bboxes()), [plask.geometry.Box2D(-2.5,0,2.5,3), plask.geometry.Box2D(0.0,3,5.0,6), plask.geometry.Box2D(-5.0,6,0.0,9)])
        stack.move_item(0, left=0)
        stack.move_item(third, left=0)
        self.assertEqual( list(stack.get_leafs_bboxes()), [plask.geometry.Box2D(0.0,0,5.0,3), plask.geometry.Box2D(0.0,3,5.0,6), plask.geometry.Box2D(0.0,6,5.0,9)])

    def testMultiStack(self):
        multistack = plask.geometry.MultiStack2D(5, 10.0)
        hint1 = multistack.append(self.block1)
        hint2 = multistack.append(self.block2)
        self.assertEqual( multistack.get_material(0.,10.), self.block1.get_material(0.,0.) )

        self.assertIn( self.block1, multistack )
        # 5 * 2 items = 10 objects, each have size 5x3, should be in [0, 10] - [5, 40]
        self.assertEqual( multistack.bbox, plask.geometry.Box2D(0., 10.0, 5., 40.0) )
        self.assertEqual( multistack.get_material(1.0, 39.0), self.aln )
        self.assertIsNone( multistack.get_material(4.0, 41.0) )
        self.assertEqual( multistack[0].item, self.block1 )
        self.assertEqual( multistack[0].vec, plask.vec(0., 10.) )
        self.assertEqual( multistack[9].item, self.block2 )
        self.assertEqual( multistack[9].vec, plask.vec(0., 37.) )
        hints1 = plask.geometry.PathHints()
        hints1 += hint1
        self.assertEqual( len(multistack[hints1]), 1 )
        self.assertEqual( multistack[hints1][0].item, self.block1 )
        self.assertEqual( multistack[hint2][0].item, self.block2 )

    def testRemoval(self):
        '''Test if removing objects from container works. In addition test prepending objects'''
        align = plask.geometry.Align2D()
        h = align.append(self.block1, 0,0) # be removed by hint
        align.append(self.block2, 10,0)
        align.append(self.block1, 10,0) # to be removed by index
        align.append(self.block2, 5,0) # to be removed by object
        self.assertEqual( align[0].item, self.block1 )
        self.assertEqual( align.get_material(12,1), self.gan)
        del align[2]
        self.assertEqual( len(align), 3 )
        self.assertEqual( align.get_material(12,1), self.aln)
        del align[h]
        self.assertEqual( len(align), 2 )
        self.assertEqual( align.get_material(1,1), None )
        del align[align[-1]]
        self.assertEqual( len(align), 1 )
        self.assertEqual( align.get_material(6,1), None )

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

    def testItemIndex(self):
        stack = plask.geometry.Stack2D()
        path0 = stack.append(self.block1)
        stack.append(self.block1)
        path2 = stack.append(self.block1)
        self.assertEqual( stack.index(self.block1, path0), 0 )
        self.assertEqual( stack.index(self.block1, path2), 2 )
        with self.assertRaises(ValueError):  # multiple instances
            stack.index(self.block1)
        with self.assertRaises(ValueError):  # no instance
            stack.index(self.block2)

    def testStackZero(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block1)
        stack.append(self.block1)
        path = stack.append(self.block1)
        self.assertEqual( stack.bbox, geometry.Box2D(0,0, 5,9) )
        stack.set_zero_below(self.block1, path)
        self.assertEqual( stack.bbox, geometry.Box2D(0,-6, 5,3) )

    def testStackZeroInObject(self):
        stack2 = plask.geometry.Stack2D()
        stack2.append(self.block1)
        p2 = stack2.append(self.block1)
        stack2.align_zero_on(self.block1, p2)
        stack1 = plask.geometry.Stack2D()
        stack1.append(stack2)
        stack1.align_zero_on(stack2, -1)
        self.assertEqual( stack2.bbox, geometry.Box2D(0,-3, 5,3) )
        self.assertEqual( stack1.bbox, geometry.Box2D(0,-2, 5,4) )

    def testRoles(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block1)
        stack.append(self.block2)
        self.assertIn( "something", stack.get_roles(2., 4.) )
        self.assertTrue( stack.has_role("something", 2., 4.) )
        self.assertEqual( stack.get_role_objects("something"), [self.block2] )

    def testGetting(self):
        stack = plask.geometry.Stack2D()
        stack.append(self.block1)
        stack.append(self.block2)
        stack.append(self.block1)
        self.assertEqual( stack.get_matching_objects(lambda o: o == self.block1), [self.block1, self.block1] )


class Edges(unittest.TestCase):

    def testSymmetricPeriodic(self):
        GaN = plask.material.GaN()
        AlN = plask.material.AlN()
        shelf = geometry.Shelf2D()
        shelf.append(geometry.Block2D(2., 2., GaN))
        shelf.append(geometry.Block2D(3., 2., AlN))
        space = geometry.Cartesian2D(shelf, left='mirror', right='periodic')
        self.assertEqual( space.get_material(1., 1.), GaN )
        self.assertEqual( space.get_material(3., 1.), AlN )
        self.assertEqual( space.get_material(4., 1.), AlN )
        self.assertEqual( space.get_material(6., 1.), AlN )
        self.assertEqual( space.get_material(7., 1.), AlN )
        self.assertEqual( space.get_material(9., 1.), GaN )
        self.assertEqual( space.get_material(11., 1.), GaN )
        self.assertEqual( space.get_material(13., 1.), AlN )
        self.assertEqual( space.get_material(-1., 1.), GaN )
        self.assertEqual( space.get_material(-3., 1.), AlN )
        self.assertEqual( space.get_material(-4., 1.), AlN )
        self.assertEqual( space.get_material(-6., 1.), AlN )
        self.assertEqual( space.get_material(-7., 1.), AlN )
        self.assertEqual( space.get_material(-9., 1.), GaN )


class ModifyObjectsTest(unittest.TestCase):

    def setUp(self):
        stack = plask.geometry.Stack2D()
        r1 = plask.geometry.Rectangle(2, 1, "GaAs")
        r2 = plask.geometry.Rectangle(2, 1, "AlAs")
        r3 = plask.geometry.Rectangle(2, 1, "InAs")
        r1.role = 'GaN'
        r2.role = 'del'
        stack.prepend(r1)
        stack.prepend(r2)
        stack.prepend(r3)
        self.geometry = plask.geometry.Cartesian2D(stack)

    def filter(self, obj):
        if isinstance(obj, (plask.geometry.GeometryObjectLeaf2D, plask.geometry.GeometryObjectLeaf3D)):
            roles = list(obj.roles)
            if 'del' in roles:
                return []
            elif roles:
                obj.material = roles[-1]
                return obj

    def testModify(self):
        geometry2 = self.geometry.modify_objects(self.filter)
        self.assertEqual( str(geometry2.get_material(1, 0.5)), 'InAs' )
        self.assertEqual( str(geometry2.get_material(1, 1.5)), 'GaN' )
        self.assertEqual( str(geometry2.get_material(1, 2.5)), 'air' )


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
