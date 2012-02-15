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
        ptest.addMyMaterial(plask.materials)


    def testMaterial(self):
        '''Test basic behavior of Material class'''
        result = plask.material.Material._completeComposition(dict(Ga=0.3, Al=None, As=0.0, N=None))
        correct = dict(Ga=0.3, Al=0.7, As=0.0, N=1.0)
        for k in correct:
            self.assertAlmostEqual( result[k], correct[k] )


    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        @plask.material.simple
        class AlGaAs(plask.material.Material):
            def __init__(self):
                super(AlGaAs, self).__init__()
                ptest.print_ptr(self)
            def __del__(self):
                ptest.print_ptr(self)
            def VBO(self, T=300.):
                return 2.*T

        self.assertIn( "AlGaAs", plask.materials )

        m = AlGaAs()
        self.assertEqual(m.name, "AlGaAs")
        self.assertEqual( m.VBO(1.0), 2.0 )
        del m

        self.assertEqual( ptest.materialName("AlGaAs", plask.materials), "AlGaAs" )
        self.assertEqual( ptest.materialVBO("AlGaAs", plask.materials, 1.0), 2.0 )

        print >>sys.stderr, plask.materials.all
        if sys.version >= 2.7:
            with self.assertRaises(RuntimeError): plask.materials.get("AlGaAs:Si=1e14")

        @plask.material.complex
        class AlGaAsDp(plask.material.Material):
            name = "AlGaAs:Dp"
            def __init__(self, *args, **kwargs):
                super(AlGaAsDp, self).__init__()
                self.args = args,
                print kwargs
                self.kwargs = kwargs
                self.composition = self._completeComposition(kwargs, self.name);
                print self.composition
            def __del__(self):
                ptest.print_ptr(self)
            def VBO(self, T=300.):
                return self.kwargs['dc'] * T
            def CBO(self, T=300.):
                return self.composition['Ga'] * T

        print plask.materials.all
        m = plask.materials.get("Al(0.2)GaAs:Dp=3.0")
        self.assertEqual( m.__class__, AlGaAsDp )
        self.assertEqual( m.name, "AlGaAs:Dp" )
        self.assertEqual( m.VBO(1.0), 3.0 )
        self.assertAlmostEqual( m.CBO(1.0), 0.8 )

        if sys.version > "2.7":
            with(self.assertRaisesRegexp(KeyError, 'N not allowed in material AlGaAs:Dp')): m = AlGaAsDp(Al=0.2, N=0.9)

        AlGaAs = lambda **kwargs: plask.materials.get("AlGaAs", **kwargs)
        m = AlGaAs(Al=0.2, dopant="Dp", dc=5.0)
        self.assertEqual( m.name, "AlGaAs:Dp" )
        self.assertEqual( m.VBO(), 1500.0 )
        correct = dict(Al=0.2, Ga=0.8, As=1.0)
        for k in correct:
            self.assertAlmostEqual( m.composition[k], correct[k] )
        del m
        self.assertEqual( ptest.materialName("Al(0.2)GaAs:Dp=3.0", plask.materials), "AlGaAs:Dp" )
        self.assertEqual( ptest.materialVBO("Al(0.2)GaAs:Dp=3.0", plask.materials, 1.0), 3.0 )

        @plask.material.simple
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
        m = plask.materials.get("MyMaterial")
        self.assertEqual( plask.materials.get("MyMaterial").name, "MyMaterial" )
        self.assertEqual( plask.materials.get("MyMaterial").VBO(1.0), 0.5 )
        self.assertEqual( ptest.materialName("MyMaterial", plask.materials), "MyMaterial" )
        self.assertEqual( ptest.materialVBO("MyMaterial", plask.materials, 1.0), 0.5 )
        self.assertEqual( ptest.call_chi(m, 'B'), 1.0)
        self.assertEqual( m.VBO(), 150.0)
        self.assertEqual( m.chi(point='C'), 1.0)


    def testMaterialWithBase(self):
        mm = plask.materials.get("MyMaterial")

        @plask.material.simple
        class WithBase(plask.material.Material):
            def __init__(self):
                super(WithBase, self).__init__(mm)

        m1 = WithBase()
        self.assertEqual( m1.name, "WithBase" )
        self.assertEqual( m1.VBO(1.0), 0.5 )

        m2 = plask.materials.get("WithBase")
        self.assertEqual( m2.name, "WithBase" )
        self.assertEqual( m2.VBO(2.0), 1.0 )

        self.assertEqual( ptest.materialName("WithBase", plask.materials), "WithBase" )
        self.assertEqual( ptest.materialVBO("WithBase", plask.materials, 1.0), 0.5 )
