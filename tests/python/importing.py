#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import sys, os


class Importing(unittest.TestCase):

    def setUp(self):
        self.builtin = 'plaskcore' in sys.builtin_module_names

    def testImporting(self):
        '''Check if plask is present'''
        print("PATH:\n   %s\n" % os.environ['PATH'])
        print("PYTHONPATH:\n")
        for p in sys.path:
            print("    %s\n" % p)
        import plask
        self.assertIn('plask.plaskcore', sys.modules)

        def use_plaskcore_directly(): print(plaskcore.__file__)
        self.assertRaises(NameError, use_plaskcore_directly)
