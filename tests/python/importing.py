#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
