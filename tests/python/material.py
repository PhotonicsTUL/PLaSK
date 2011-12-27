#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import plask.material

class Material(unittest.TestCase):

    def setUp(self):
        pass

    def testMaterial(self):
        '''Test basic behavior of Material class'''
        self.assertRaises(RuntimeError, lambda: plask.material.Material())

        #self.assertRaises(TypeError, random.shuffle, (1,2,3))
        #self.assertEqual(self.seq, range(10))
        #self.assertTrue(element in self.seq)

    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        @plask.material.new
        class Mat(plask.material.CustomMaterial):
            pass

        m = Mat()

        self.assertEqual(m.getName(), "Mat")
        self.assertEqual(m.name, "Mat")
