#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

import plask
import plask.mesh
import plask.geometry

class DataTest(unittest.TestCase):

    def testFromArray(self):
        m1 = plask.mesh.Rectangular2D([0, 1], [0, 1, 2], ordering='01')
        m2 = plask.mesh.Rectangular2D([0, 1], [0, 1, 2], ordering='10')
        a = array([[0., 1., 2.], [3., 4., 5.]])
        d1 = plask.Data(a, m1)
        d2 = plask.Data(a, m2)
        self.assertEqual( list(d1), [0., 1., 2., 3., 4., 5.] )
        self.assertEqual( list(d2), [0., 3., 1., 4., 2., 5.] )
        self.assertTrue( (d1.array == d2.array).all() )
