#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

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
        pass

    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        @plask.material.new
        class Mat(plask.material.Material):
            dopants = ['Si', 'Ge']

            def __init__(self, *args, **kwargs):
                super(Mat, self).__init__()
                self.args = args,
                self.kwargs = kwargs;

            def VBO(self, T):
                return 2.*T

        m = Mat()
        self.assertEqual(m.name, "Mat")
        self.assertEqual( m.VBO(1.0), 2.0 )
        del m

        self.assertIn( "Mat", self.DB )

        self.assertEqual( ptest.materialName("Mat", plask.material.database), "Mat" )
        self.assertEqual( ptest.materialVBO("Mat", plask.material.database, 1.0), 2.0 )

        m = self.DB.get("Mat:Si=1e17")
        self.assertEqual( m.name, "Mat" )
        del m
        self.assertEqual( ptest.materialName("Mat:Si=1e16", plask.material.database), "Mat" )
        self.assertEqual( ptest.materialName("Mat:Ge=1e15", plask.material.database), "Mat" )
        with self.assertRaises(RuntimeError): self.DB.get("Mat:Mg=1e14")


        @plask.material.new (name="Mat:Dp")
        class MatDp(plask.material.Material):

            def __init__(self, *args, **kwargs):
                super(MatDp, self).__init__()
                print >> sys.stderr, "args:", args
                print >> sys.stderr, "kwargs:", kwargs
                self.args = args,
                self.kwargs = kwargs;

            def VBO(self, T):
                return self.kwargs['dc'] * T

        m = self.DB.get("Mat:Dp=3.0")
        self.assertEqual( m.name, "Mat:Dp" )
        self.assertEqual( m.VBO(1.0), 3.0 )

        Mat = lambda *args, **kwargs: self.DB.factory("Mat", args, kwargs)
        m = Mat(dope="Dp", dc=5.0)
        self.assertEqual( m.name, "Mat:Dp" )
        self.assertEqual( m.VBO(1.0), 5.0 )
        del m

        self.assertEqual( ptest.materialName("Mat:Dp=3.0", plask.material.database), "Mat:Dp" )
        self.assertEqual( ptest.materialVBO("Mat:Dp=3.0", plask.material.database, 1.0), 3.0 )


    def testExistingMaterial(self):
        '''Test if existing material works correctly'''
        self.assertEqual( ptest.materialName("MyMaterial", plask.material.database), "MyMaterial" )
        self.assertEqual( ptest.materialVBO("MyMaterial", plask.material.database, 1.0), 0.5 )
        self.assertEqual( self.DB.factory("MyMaterial", (), {}).name, "MyMaterial" )
        self.assertEqual( self.DB.factory("MyMaterial", (), {}).VBO(1.0), 0.5 )
        self.assertEqual( self.DB.get("MyMaterial").name, "MyMaterial" )
        self.assertEqual( self.DB.get("MyMaterial").VBO(1.0), 0.5 )
