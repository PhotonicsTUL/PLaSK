#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import plask

class Vector(unittest.TestCase):

    def setUp(self):
        self.a2 = plask.vector(1,2)
        self.b2 = plask.vector(10,20, float)
        self.c2 = plask.vector(100,200, complex)
        self.a3 = plask.vector(1,2,3)
        self.b3 = plask.vector(10,20,30, float)
        self.c3 = plask.vector(100,200,300, complex)

    def testFactory(self):
        self.assertEqual( type(self.a2), plask.vector2d_float )
        self.assertEqual( type(self.b2), plask.vector2d_float )
        self.assertEqual( type(self.c2), plask.vector2d_complex )
        self.assertEqual( type(self.a3), plask.vector3d_float )
        self.assertEqual( type(self.b3), plask.vector3d_float )
        self.assertEqual( type(self.c3), plask.vector3d_complex )

    def testItemAccess(self):
        '''Test existence and properties of classes'''
        self.assertEqual( [self.a2.x, self.a2.y], [1,2] )
        self.assertEqual( [self.a2.r, self.a2.z], [1,2] )
        self.assertEqual( [self.a2[0], self.a2[1]], [1,2] )

    def testOperations(self):
        self.assertEqual( self.a2.magnitude2(), 5 )
        self.assertTrue ( self.a2 != self.c2)
        self.assertTrue ( self.a2 == plask.vector(1,2) )
        self.assertEqual( self.a2 + self.c2, plask.vector(101,202) )
        self.assertEqual( self.a2.dot(self.b2), 50)
        self.assertEqual( -self.a2, plask.vector(-1,-2) )
        self.assertEqual( 2 * self.a2, plask.vector(2, 4) )
        self.assertEqual( 10 * self.b3, self.c3 )
        self.assertEqual( self.c3, 10 * self.b3 )
        self.a2 *= 2
        self.a2 += self.b2
        self.assertEqual( self.a2, plask.vector(12,24) )
        self.a2 -= self.b2
        self.a2 *= 0.5
        self.assertEqual( self.a2, plask.vector(1,2) )
        self.assertEqual( self.b3 * 0.5, plask.vector(5, 10, 15) )