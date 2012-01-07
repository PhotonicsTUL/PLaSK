#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import plask, plask.material, plask.geometry

class Geometry(unittest.TestCase):

    def setUp(self):
        pass

    def testClasses(self):
        '''Test existence and properties of classes'''
        pass

    def testMemoryManagement(self):
        '''Test if the objects live in memory as long as needed'''
        pass

    def testPrimitives(self):
        '''Test the properties of primitives'''
        r2a = plask.geometry.Rect2D()
        r2a.lower = plask.vec(3., 2.)
        r2a.upper = plask.vec(1., 5.)
        r2a.fix()
        self.assertEqual( r2a.lower, plask.vec(1,2) )
        self.assertEqual( r2a.upper, plask.vec(3,5) )
        r2b = plask.geometry.Rect2D(plask.vec(3., 2.), plask.vec(1., 5.))
        r2b.fix()
        self.assertEqual( r2b.lower, plask.vec(1,2) )
        self.assertEqual( r2b.upper, plask.vec(3,5) )
        r3a = plask.geometry.Rect3D(3.,2.,1., 1.,5.,0.)
        r3b = plask.geometry.Rect3D(plask.vec(1.,2.,0.), plask.vec(3.,5.,1.))
        self.assertEqual( r3a, r3b )

    def testPathHints(self):
        '''Test if path hints work'''