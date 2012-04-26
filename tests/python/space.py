#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import sys
if sys.version < "2.7":
    unittest.TestCase.assertIsNone = lambda self, value: self.assertTrue(item is None)
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)

import plask, plask.materials, plask.geometry



class CalculationSpaces(unittest.TestCase):

    def setUp(self):
        pass

    def testSpace2DCartesian(self):
        r = plask.geometry.Rectangle(2,1, "GaN")
        s = plask.Space2DCartesian(r,