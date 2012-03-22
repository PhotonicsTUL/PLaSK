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

    def testSwapped(self):
        self.assertEqual( list(self.mesh2), [plask.vec(1,10), plask.vec(2,10), plask.vec(3,10), plask.vec(1,20), plask.vec(2,20), plask.vec(3,20)] )

        swapped = ptest.swapMesh(self.mesh2)
        self.assertEqual( list(swapped), [plask.vec(1,10), plask.vec(1,20), plask.vec(2,10), plask.vec(2,20), plask.vec(3,10), plask.vec(3,20)] )

        reswapped = ptest.swapMesh(swapped)
        self.assertEqual( list(reswapped), [plask.vec(1,10), plask.vec(2,10), plask.vec(3,10), plask.vec(1,20), plask.vec(2,20), plask.vec(3,20)] )

        self.assertEqual( list(self.mesh2.makeOptimized()), [plask.vec(1,10), plask.vec(1,20), plask.vec(2,10), plask.vec(2,20), plask.vec(3,10), plask.vec(3,20)] )

