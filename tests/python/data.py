#!/usr/bin/env plask
# -*- coding: utf-8 -*-
import unittest
import sys

from numpy import *
from numpy.testing import assert_array_equal

import plask
from plask import Data, mesh

class DataTest(unittest.TestCase):

    def testFromArray(self):
        m1 = mesh.Rectangular2D([0, 1], [0, 1, 2], ordering='01')
        m2 = mesh.Rectangular2D([0, 1], [0, 1, 2], ordering='10')
        a = array([[0., 1., 2.], [3., 4., 5.]])
        d1 = Data(a, m1)
        d2 = Data(a, m2)
        self.assertEqual(list(d1), [0., 1., 2., 3., 4., 5.])
        self.assertEqual(list(d2), [0., 3., 1., 4., 2., 5.])
        assert_array_equal(d1.array, d2.array)

    def testMultiIndex(self):
        data2d = Data(array([[1., 2.], [3., 4.]]), mesh.Rectangular2D([0., 1.], [0., 1.]))
        self.assertIsInstance(data2d[0, 1], float)
        self.assertEqual(data2d[0, 1], 2.)

        data3d = Data(array([[[1., 2.], [3., 4.]]]), mesh.Rectangular3D([0.], [0., 1.], [0., 1.]))
        self.assertIsInstance(data3d[0, 0, 1], float)
        self.assertEqual(data3d[0, 1, -1], 4.)

    def testSlice2D(self):
        axis = mesh.Regular(0, 9, 10)
        a = array(axis)
        arr = 10 * a[:,None] + a[None,:]
        data = Data(arr, mesh.Rectangular2D(axis, axis))

        data1 = data[1:4, 0:-2:4]
        self.assertEqual(data1.mesh.axis0, [1, 2, 3])
        self.assertEqual(data1.mesh.axis1, [0, 4])
        assert_array_equal(data1.array, arr[1:4, 0:-2:4])

        data2 = data[5,:]
        self.assertEqual(data2.mesh.axis0, [5])
        self.assertEqual(data2.mesh.axis1, axis)
        assert_array_equal(data2.array, arr[[5],:])

        with self.assertRaises(TypeError):
            data[:,:,:]

    def testSlice3D(self):
        axis = mesh.Regular(0, 9, 10)
        a = array(axis)
        arr = 100 * a[:,None,None] + 10 * a[None,:,None] + a[None,None,:]
        data = Data(arr, mesh.Rectangular3D(axis, axis, axis))

        data1 = data[1:4, 0::4, -5:-1]
        self.assertEqual(data1.mesh.axis0, [1, 2, 3])
        self.assertEqual(data1.mesh.axis1, [0, 4, 8])
        self.assertEqual(data1.mesh.axis2, [5, 6, 7, 8])
        assert_array_equal(data1.array, arr[1:4, 0::4, -5:-1])

        with self.assertRaises(TypeError):
            data[:,:]


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
