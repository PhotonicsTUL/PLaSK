#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import sys, os

class ConstTempImporting(unittest.TestCase):

    def setUp(self):
        self.builtin = 'plaskcore' in sys.builtin_module_names

    def testImporting(self):
        '''Check if module can be imported is present'''
        from plask.thermal.const import ConstantTemperature
        mod = ConstantTemperature()
        self.assertEqual(mod.name, "Constant Temperature")
