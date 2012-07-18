#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

import plask.mesh
import plasktest




class RectilinearMeshes(unittest.TestCase):

    def setUp(self):
        self.mesh2 = plask.mesh.Rectilinear2D([1,3,2,1], array([10,20], float))
        self.mesh3 = plask.mesh.Rectilinear3D([1,2,3], [10,20], [100,200])

    def testOrdering2D(self):
        m = self.mesh2

        self.assertEqual( [list(i) for i in m], [[1,10], [2,10], [3,10], [1,20], [2,20], [3,20]] )

        m.setOrdering("01")
        self.assertEqual( [list(i) for i in m], [[1,10], [1,20], [2,10], [2,20], [3,10], [3,20]] )
        self.assertEqual( m.minor_axis, m.axis1 )
        self.assertEqual( m.major_axis, m.axis0 )

        m.setOrdering("10")
        self.assertEqual( [list(i) for i in m], [[1,10], [2,10], [3,10], [1,20], [2,20], [3,20]] )

        m.setOptimalOrdering()
        self.assertEqual( [list(i) for i in m], [[1,10], [1,20], [2,10], [2,20], [3,10], [3,20]] )

        for i in range(len(m)):
            i0 = m.index0(i)
            i1 = m.index1(i)
            self.assertEqual( m.index(i0, i1), i )


    def testOrdering3D(self):
        m = self.mesh3

        self.assertEqual( [list(i) for i in m], [[1,10,100], [2,10,100], [3,10,100], [1,20,100], [2,20,100], [3,20,100],
                                                 [1,10,200], [2,10,200], [3,10,200], [1,20,200], [2,20,200], [3,20,200]] )
        m.setOrdering('201')
        self.assertEqual( [list(i) for i in m], [[1,10,100], [1,20,100], [2,10,100], [2,20,100], [3,10,100], [3,20,100],
                                                 [1,10,200], [1,20,200], [2,10,200], [2,20,200], [3,10,200], [3,20,200]] )

        for i in range(len(m)):
            i0 = m.index0(i)
            i1 = m.index1(i)
            i2 = m.index2(i)
            self.assertEqual( m.index(i0, i1, i2), i )

        m.setOptimalOrdering()
        self.assertEqual( [list(i) for i in m], [[1,10,100], [1,10,200], [1,20,100], [1,20,200],
                                                [2,10,100], [2,10,200], [2,20,100], [2,20,200],
                                                [3,10,100], [3,10,200], [3,20,100], [3,20,200]] )
        self.assertEqual( m.minor_axis,  m.axis2 )
        self.assertEqual( m.middle_axis, m.axis1 )
        self.assertEqual( m.major_axis,  m.axis0 )

    def testBoundary(self):
        self.mesh2.setOrdering("10")
        b = plask.mesh.Rectilinear2D.leftBoundary(self.mesh2)
        self.assertIn(0, b)
        self.assertNotIn(1, b)
        self.assertIn(3, b)
        self.assertEqual( list(b), [0, 3] )

        self.assertEqual( plasktest.testBoundary(self.mesh2, lambda mesh,i: i in [2,3]), [2, 3] )


    def testDividingGenerator(self):
        stack = plask.geometry.Stack2D()
        stack.append(plask.geometry.Rectangle(2, 2, None))
        stack.append(plask.geometry.Rectangle(2, 16, None))
        rect = plask.geometry.Rectangle(2, 16, None)
        stack.append(rect)

        generator1 = plask.mesh.Rectilinear2D.SimpleGenerator()
        generator2 = plask.mesh.Rectilinear2D.DividingGenerator()
        generator2.prediv = 2
        generator2.addRefinement("z", rect, 8.)

        self.assertEqual( list(generator2.getRefinements("z").values()), [[8.]] )

        mesh1 = generator1(stack)
        self.assertEqual( list(mesh1.axis0), [0., 2.] )
        self.assertEqual( list(mesh1.axis1), [0., 2., 18., 34.] )

        mesh2 = generator2(stack)
        self.assertEqual( list(mesh2.axis0), [0., 1., 2.] )
        self.assertEqual( list(mesh2.axis1), [0., 1., 2., 4., 6., 10., 18., 22., 26., 30., 34.] )
