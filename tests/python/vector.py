#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy

import plask
import plasktest

class Vector(unittest.TestCase):

    def setUp(self):
        self.a2 = plask.vec(1,2)
        self.b2 = plask.vec(10,20)
        self.c2 = plask.vec(100,200, dtype=complex)
        self.d2 = plask.vec(1+2j, 0)
        self.a3 = plask.vec(1,2,3)
        self.b3 = plask.vec(10,20,30)
        self.c3 = plask.vec(100,200,300, dtype=complex)
        self.d3 = plask.vec(1+2j, 0, 0)

    def testTypes(self):
        '''Test if dtypes are set correctly'''
        self.assertEqual( self.a2.dtype, float )
        self.assertEqual( self.b2.dtype, float )
        self.assertEqual( self.c2.dtype, complex )
        self.assertEqual( self.d2.dtype, complex )

    def testFactory(self):
        '''Test vector creation by component names'''
        va = plask.config.vertical_axis

        plask.config.vertical_axis = 'z'
        self.assertAlmostEqual( plask.vec(y=1, z=2), plask.vec(1,2) )
        self.assertAlmostEqual( plask.vec(r=1, z=2), plask.vec(1,2) )
        self.assertAlmostEqual( plask.vec(z=3, x=1, y=2), plask.vec(1,2,3) )
        self.assertAlmostEqual( plask.vec(r=1, z=3, phi=2), plask.vec(1,2,3) )
        self.assertRaises( TypeError, lambda: plask.vec(x=1, y=2) ) # for vertical_axis = 'z' x component is not allowed in 2D
        self.assertRaises( TypeError, lambda: plask.vec(bad_x=1, z=2, y=1) )
        self.assertRaises( TypeError, lambda: plask.vec(r=1, y=2, z=3) )

        plask.config.vertical_axis = 'y'
        self.assertAlmostEqual( plask.vec(x=1, y=2), plask.vec(1,2) )
        self.assertAlmostEqual( plask.vec(y=3, x=2, z=1), plask.vec(1,2,3) )
        self.assertRaises( TypeError, lambda: plask.vec(y=1, z=2) ) # for vertical_axis = 'y' z component is not allowed in 2D
        self.assertRaises( TypeError, lambda: plask.vec(r=1, z=2) )
        self.assertRaises( TypeError, lambda: plask.vec(phi=1, y=2, z=3) )
        self.assertRaises( TypeError, lambda: plask.vec(x=1, bad_x=2) )

        plask.config.vertical_axis = va

    def testItemAccess(self):
        '''Test if the items can be accessed corretly using all possible ways'''
        self.assertAlmostEqual( [self.a2[0], self.a2[1]], [1,2] )
        self.assertAlmostEqual( [self.a3[-3], self.a3[-2], self.a3[-1]], [1,2,3] )

        self.c3[0] = 1j
        self.assertAlmostEqual( self.c3, plask.vec(1j,200,300) )
        self.c3[0] = 100.

        a = plask.vec(1,2, dtype=complex)
        a[1] = 2j
        self.assertAlmostEqual( a, plask.vec(1,2j) )
        a = plask.vec(1,2,3, dtype=complex)
        a[1] = 2j
        self.assertAlmostEqual( a, plask.vec(1,2j,3) )

        va = plask.config.vertical_axis

        plask.config.vertical_axis = 'z'
        self.assertEqual( [self.a2.y, self.a2.z], [1, 2] )
        self.assertEqual( [self.a2.r, self.a2.z], [1, 2] )
        self.assertEqual( [self.a3.x, self.a3.y, self.a3.z], [1, 2, 3] )
        self.assertEqual( [self.a3.r, self.a3.phi, self.a3.z], [1, 2, 3] )

        plask.config.vertical_axis = 'y'
        self.assertEqual( [self.a2.x, self.a2.y], [1, 2] )
        self.assertEqual( [self.a3.z, self.a3.x, self.a3.y], [1, 2, 3] )

        self.c3.x = 2j
        self.assertEqual( [self.c3.z, self.c3.x, self.c3.y], [100, 2j, 300] )
        self.c3.x = 200

        plask.config.vertical_axis = va

    def testArray(self):
        a = numpy.array(self.c3)
        self.assertEqual( list(a), [100, 200, 300] )
        many = plasktest.getVecs()
        arr = numpy.array(many, copy=False)
        self.assertEqual( sys.getrefcount(many), 3 ) # many, arr, getrefcount
        self.assertEqual( list(arr.shape), [5, 2] )
        del many
        self.assertEqual( arr[3,0], 7 )


    def testExceptions(self):
        '''Test if proper exceptions are thrown'''
        self.assertRaises( TypeError, lambda: plask.vec(1,2j, dtype=float) )
        self.assertRaises( TypeError, lambda: plask.vec(1,2,'a') )
        self.assertRaises( TypeError, lambda: plask.vec(1,2,z=3) )
        self.assertRaises( IndexError, lambda: self.a2[2] )
        self.assertRaises( IndexError, lambda: self.a3[3] )
        self.assertRaises( IndexError, lambda: self.a2[-3] )
        self.assertRaises( IndexError, lambda: self.a3[-4] )

    def testOperations(self):
        '''Test vector mathematical operations'''
        self.assertAlmostEqual( self.c2, plask.vec(100, 200) )
        self.assertAlmostEqual( plask.vec(100, 200, 300), self.c3 )
        self.assertAlmostEqual( self.a2.abs2(), 5 )
        self.assertAlmostEqual( abs(self.a3)**2, 14 )
        self.assertTrue ( self.a2 != self.c2)
        #self.assertTrue ( self.a2 == plask.vec(1,2) )
        self.assertAlmostEqual( self.a2 + self.c2, plask.vec(101,202) )
        self.assertAlmostEqual( self.a2.dot(self.b2), 50)
        self.assertAlmostEqual( self.a2.dot(self.c2), 500)
        self.assertAlmostEqual( self.c2.dot(self.a2), 500)
        self.assertAlmostEqual( self.a2 * self.b2, 50)
        self.assertAlmostEqual( self.a2 * self.c2, 500)
        self.assertAlmostEqual( self.c2 * self.a2, 500)
        self.assertAlmostEqual( -self.a2, plask.vec(-1,-2) )
        self.assertAlmostEqual( 2 * self.a2, plask.vec(2, 4) )
        self.assertAlmostEqual( 10 * self.b3, self.c3 )
        self.assertAlmostEqual( self.b3 * 10, self.c3 )
        self.assertAlmostEqual( self.c3, 10 * self.b3 )
        self.a2 *= 2
        self.a2 += self.b2
        self.assertAlmostEqual( self.a2, plask.vec(12,24) )
        self.a2 -= self.b2
        self.a2 *= 0.5
        self.assertAlmostEqual( self.a2, plask.vec(1,2) )
        self.assertAlmostEqual( self.b3 * 0.5, plask.vec(5, 10, 15) )
        self.assertAlmostEqual( self.a2.conj(), self.a2 )
        self.assertAlmostEqual( self.d2.conj(), plask.vec(1-2j, 0) )
        self.assertAlmostEqual( self.d3.conj(), plask.vec(1-2j, 0, 0) )
        self.assertAlmostEqual( list(self.a3), [1,2,3] )
        pass
