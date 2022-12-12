#!/usr/bin/env plask
# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import sys
import unittest

import sys, os
if sys.version < "2.7":
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)


class TestImporting(unittest.TestCase):

    def setUp(self):
        self.builtin = '_plask' in sys.builtin_module_names

    def testImporting(self):
        '''Check if plask is present'''
        print("PATH:\n   %s" % os.environ['PATH'])
        print("PYTHONPATH:")
        for p in sys.path:
            print("    %s" % p)
        import plask
        self.assertIn('plask._plask', sys.modules)

        with self.assertRaises(NameError):
            print(_plask.__file__)

        with self.assertRaises(AttributeError):
            print(plask._plask.__file__)

    def testAutoImported(self):
        '''Test if the plask is auto-imported to global namespace for plask binary'''
        if self.builtin:
            plask.version
            version

    def testDivision(self):
        '''Test if the plask binary uses correct division
           (we test only for built-in as with the module we would need to import division explicitly)'''
        if self.builtin:
            self.assertEqual(1/2, 0.5)
    pass


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
