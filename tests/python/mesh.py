#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

import plasktest as ptest

import sys
if sys.version < "2.7":
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)

import plask.meshes



class Meshes(unittest.TestCase):

    def setUp(self):
        self.mesh2 = plask.meshes.Rectilinear2D(plask.meshes.Rectilinear1D([1,2,3]), plask.meshes.Rectilinear1D([10,20]))
        self.mesh3 = plask.meshes.Rectilinear3D(plask.meshes.Rectilinear1D([1,2,3]), plask.meshes.Rectilinear1D([10,20]), plask.meshes.Rectilinear1D([100,200]))


    def testOrdering2D(self):
        m = self.mesh2

        self.assertEqual( map(list, m), [[1,10], [2,10], [3,10], [1,20], [2,20], [3,20]] )

        m.setOrdering("10")
        self.assertEqual( map(list, m), [[1,10], [1,20], [2,10], [2,20], [3,10], [3,20]] )

        m.setOrdering("01")
        self.assertEqual( map(list, m), [[1,10], [2,10], [3,10], [1,20], [2,20], [3,20]] )

        m.setOptimalOrdering()
        self.assertEqual( map(list, m), [[1,10], [1,20], [2,10], [2,20], [3,10], [3,20]] )

        for i in range(len(m)):
            i0 = m.index0(i)
            i1 = m.index1(i)
            self.assertEqual( m.index(i0, i1), i )


    def testOrdering3D(self):
        m = self.mesh3

        m.setOrdering('102')
        self.assertEqual( map(list,m), [[1,10,100], [1,20,100], [2,10,100], [2,20,100], [3,10,100], [3,20,100],
                                        [1,10,200], [1,20,200], [2,10,200], [2,20,200], [3,10,200], [3,20,200]] )

        for i in range(len(m)):
            i0 = m.index0(i)
            i1 = m.index1(i)
            i2 = m.index2(i)
            self.assertEqual( m.index(i0, i1, i2), i )

        m.setOptimalOrdering()
        self.assertEqual( map(list,m), [[1,10,100], [1,20,100], [1,10,200], [1,20,200],
                                        [2,10,100], [2,20,100], [2,10,200], [2,20,200],
                                        [3,10,100], [3,20,100], [3,10,200], [3,20,200]] )
