#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import sys
if sys.version < "2.7":
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)

import plask.material

class Material(unittest.TestCase):

    def setUp(self):
        self.DB = plask.material.database

    def testMaterial(self):
        '''Test basic behavior of Material class'''
        pass

    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        @plask.material.new
        class Mat(plask.material.Material):
            pass
        m = Mat()

        self.assertEqual(m.name(), "Mat")

        self.assertIn( "Mat", self.DB )
