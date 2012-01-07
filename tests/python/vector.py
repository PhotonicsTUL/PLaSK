#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import plask

class Vector(unittest.TestCase):

    def setUp(self):
        pass
        self.a2 = plask.vec(1,2)
        self.b2 = plask.vec(10,20, dtype=float)
        self.c2 = plask.vec(100,200, dtype=complex)
        self.d2 = plask.vec(1+2j, 0)
        self.a3 = plask.vec(1,2,3)
        self.b3 = plask.vec(10,20,30, dtype=float)
        self.c3 = plask.vec(100,200,300, dtype=complex)
        self.d3 = plask.vec(1+2j, 0, 0)

    def testFactory(self):
        '''Test automatic choice of proper class in vector factory'''
        self.assertEqual( self.a2.dtype, float )
        self.assertEqual( self.b2.dtype, float )
        self.assertEqual( self.c2.dtype, complex )
        self.assertEqual( self.d2.dtype, complex )
        self.assertEqual( self.a3.dtype, float )
        self.assertEqual( self.b3.dtype, float )
        self.assertEqual( self.c3.dtype, complex )
        self.assertEqual( self.d3.dtype, complex )

    def testItemAccess(self):
        '''Test if the items can be accessed corretly using all possible ways'''
        #self.assertEqual( [self.a2.x, self.a2.y], [1,2] )
        #self.assertEqual( [self.a2.r, self.a2.z], [1,2] )
        #self.assertEqual( [self.a2[0], self.a2[1]], [1,2] )
        #self.assertEqual( [self.a3[-3], self.a3[-2], self.a3[-1]], [1,2,3] )
        #self.assertEqual( [self.a3.r, self.a3.phi, self.a3.z], [1,2,3] )
        pass

    def testExceptions(self):
        '''Test if proper exceptions are thrown'''
        self.assertRaises( TypeError, lambda: plask.vec(1,2,z=3) )
        self.assertRaises( TypeError, lambda: plask.vec(1,2,3+1j, dtype=float) )
        #self.assertRaises( IndexError, lambda: self.a2[2] )
        #self.assertRaises( IndexError, lambda: self.a3[3] )
        #self.assertRaises( IndexError, lambda: self.a2[-3] )
        #self.assertRaises( IndexError, lambda: self.a3[-4] )

    def testOperations(self):
        '''Test vector mathematical operations'''
        self.assertTrue( self.c2 )
        self.assertTrue( not plask.vec(0,0e-30))
        self.assertTrue( not plask.vec(0,0,0))
        self.assertTrue( not plask.vec(0,0,0j))
        self.assertEqual( self.a2.abs2(), 5 )
        self.assertTrue ( self.a2 != self.c2)
        self.assertTrue ( self.a2 == plask.vec(1,2) )
        #self.assertEqual( self.a2 + self.c2, plask.vec(101,202) )
        #self.assertEqual( self.a2.dot(self.b2), 50)
        #self.assertEqual( self.a2.dot(self.c2), 500)
        #self.assertEqual( self.c2.dot(self.a2), 500)
        #self.assertEqual( self.a2 * self.b2, 50)
        #self.assertEqual( self.a2 * self.c2, 500)
        #self.assertEqual( self.c2 * self.a2, 500)
        #self.assertEqual( -self.a2, plask.vec(-1,-2) )
        #self.assertEqual( 2 * self.a2, plask.vec(2, 4) )
        #self.assertEqual( 10 * self.b3, self.c3 )
        #self.assertEqual( self.b3 * 10, self.c3 )
        #self.assertEqual( self.c3, 10 * self.b3 )
        #self.a2 *= 2
        #self.a2 += self.b2
        #self.assertEqual( self.a2, plask.vec(12,24) )
        #self.a2 -= self.b2
        #self.a2 *= 0.5
        #self.assertEqual( self.a2, plask.vec(1,2) )
        #self.assertEqual( self.b3 * 0.5, plask.vec(5, 10, 15) )
        #self.assertEqual( self.a2.conj(), self.a2 )
        #self.assertEqual( self.d2.conj(), plask.vec(1-2j, 0) )
        #self.assertEqual( self.d3.conj(), plask.vec(1-2j, 0, 0) )
        #self.assertEqual( list(self.a3), [1,2,3] )
        pass