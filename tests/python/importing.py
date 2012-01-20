#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import sys, os
if sys.version < "2.7":
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)


class Importing(unittest.TestCase):

    def setUp(self):
        self.builtin = 'modplask' in sys.builtin_module_names

    def testImporting(self):
        '''Check if plask is present'''
        print >>sys.stderr, "PATH:\n   ", os.environ['PATH']
        print >>sys.stderr, "PYTHONPATH:"
        for p in sys.path:
            print >>sys.stderr, "    %s" % p
        import plask
        self.assertIn('modplask', sys.modules)

        def use_modplask_directly(): print modplask.__file__
        self.assertRaises(NameError, use_modplask_directly)

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
