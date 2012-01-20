#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

import plasktest as ptest

import sys
if sys.version < "2.7":
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)

import plask.material

class Material(unittest.TestCase):

    def setUp(self):
        self.DB = plask.material.database
        ptest.addMyMaterial(self.DB)


    def testMaterial(self):
        '''Test basic behavior of Material class'''
        self.assertEqual( plask.material.Material._completeComposition([0.6, nan, 0.1, nan, 0.2], 23), [0.6, 0.4, 0.1, 0.7, 0.2] )


    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        @plask.material.new
        class MyMat(plask.material.Material):
            def __init__(self):
                super(MyMat, self).__init__()
                print >>sys.stderr, "MyMat.__init__",
                ptest.print_ptr(self)
            def __del__(self):
                print >>sys.stderr, "MyMat.__del__",
                ptest.print_ptr(self)
            def VBO(self, T):
                return 2.*T

        self.assertIn( "MyMat", self.DB )

        m = MyMat()
        self.assertEqual(m.name, "MyMat")
        self.assertEqual( m.VBO(1.0), 2.0 )
        del m

        self.assertEqual( ptest.materialName("MyMat", plask.material.database), "MyMat" )
        self.assertEqual( ptest.materialVBO("MyMat", plask.material.database, 1.0), 2.0 )

        if sys.version >= 2.7:
            with self.assertRaises(RuntimeError): self.DB.get("MyMat:Si=1e14")


        @plask.material.new
        class MyMatDp(plask.material.Material):
            name = "MyMat:Dp"
            def __init__(self, *args, **kwargs):
                super(MyMatDp, self).__init__()
                self.args = args,
                self.kwargs = kwargs;
                print >>sys.stderr, "MyMat:Dp.__init__",
                ptest.print_ptr(self)
            def __del__(self):
                print >>sys.stderr, "MyMat:Dp.__del__",
                ptest.print_ptr(self)
            def VBO(self, T):
                return self.kwargs['dc'] * T

        m = self.DB.get("MyMat:Dp=3.0")
        self.assertEqual( m.__class__, MyMatDp )
        self.assertEqual( m.name, "MyMat:Dp" )
        self.assertEqual( m.VBO(1.0), 3.0 )

        MyMat = lambda **kwargs: self.DB.get("MyMat", **kwargs)
        m = MyMat(Mat=0.2, dope="Dp", dc=5.0)
        self.assertEqual( m.name, "MyMat:Dp" )
        self.assertEqual( m.VBO(1.0), 5.0 )
        del m
        self.assertEqual( ptest.materialName("MyMat:Dp=3.0", plask.material.database), "MyMat:Dp" )
        self.assertEqual( ptest.materialVBO("MyMat:Dp=3.0", plask.material.database, 1.0), 3.0 )


        @plask.material.new
        class WithChar(plask.material.Material):
            def chi(self, T, p):
                print >>sys.stderr, "WithChar:", p
                return 1.5

        c = WithChar()
        self.assertEqual( c.name, "WithChar" )
        if sys.version >= 2.7:
            with self.assertRaisesRegexp(RuntimeError, "Method not implemented"): c.VBO(1.0)
        self.assertEqual( ptest.call_chi(c, 'A'), 1.5)

        #self.assertTrue(False)


    def testExistingMaterial(self):
        '''Test if existing material works correctly'''
        m = self.DB.get("MyMaterial")
        self.assertEqual( self.DB.get("MyMaterial").name, "MyMaterial" )
        self.assertEqual( self.DB.get("MyMaterial").VBO(1.0), 0.5 )
        self.assertEqual( ptest.materialName("MyMaterial", self.DB), "MyMaterial" )
        self.assertEqual( ptest.materialVBO("MyMaterial", self.DB, 1.0), 0.5 )
        self.assertEqual( ptest.call_chi(m, 'B'), 1.0)
        self.assertEqual( m.chi('C'), 1.0)


    def testMaterialWithBase(self):
        mm = self.DB.get("MyMaterial")

        @plask.material.new
        class WithBase(plask.material.Material):
            def __init__(self):
                super(WithBase, self).__init__(mm)

        m1 = WithBase()
        self.assertEqual( m1.name, "WithBase" )
        self.assertEqual( m1.VBO(1.0), 0.5 )

        m2 = self.DB.get("WithBase")
        self.assertEqual( m2.name, "WithBase" )
        self.assertEqual( m2.VBO(2.0), 1.0 )

        self.assertEqual( ptest.materialName("WithBase", self.DB), "WithBase" )
        self.assertEqual( ptest.materialVBO("WithBase", self.DB, 1.0), 0.5 )
