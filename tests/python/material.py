#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import plask.material

class Material(unittest.TestCase):

    def setUp(self):
        pass

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
