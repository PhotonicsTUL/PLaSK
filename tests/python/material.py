#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

import plasktest as ptest

import sys
if sys.version < "2.7":
    unittest.TestCase.assertIn = lambda self, item, container: self.assertTrue(item in container)

import plask.materials

class Material(unittest.TestCase):

    @plask.materials.complex
    class AlGaAs(plask.materials.Material):
        def __init__(self, **kwargs):
            super(Material.AlGaAs, self).__init__()
            ptest.print_ptr(self)
        def __del__(self):
            ptest.print_ptr(self)
        def VBO(self, T=300.):
            return 2.*T

    @plask.materials.complex
    class AlGaAsDp(plask.materials.Material):
        name = "AlGaAs:Dp"
        def __init__(self, *args, **kwargs):
            super(Material.AlGaAsDp, self).__init__()
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

    @plask.materials.simple
    class WithChar(plask.materials.Material):
        def chi(self, T, p):
            print >>sys.stderr, "WithChar:", p
            return 1.5

    def setUp(self):
        ptest.addMyMaterial(plask.materialsdb)


    def testMaterial(self):
        '''Test basic behavior of Material class'''
        result = plask.materials.Material._completeComposition(dict(Ga=0.3, Al=None, As=0.0, N=None))
        correct = dict(Ga=0.3, Al=0.7, As=0.0, N=1.0)
        for k in correct:
            self.assertAlmostEqual( result[k], correct[k] )


    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        self.assertIn( "AlGaAs", plask.materialsdb )

        m = Material.AlGaAs()
        self.assertEqual(m.name, "AlGaAs")
        self.assertEqual( m.VBO(1.0), 2.0 )
        del m

        self.assertEqual( ptest.materialName("Al(0.2)GaAs", plask.materialsdb), "AlGaAs" )
        self.assertEqual( ptest.materialVBO("Al(0.2)GaAs", plask.materialsdb, 1.0), 2.0 )

        print >>sys.stderr, plask.materialsdb.all
        if sys.version >= 2.7:
            with self.assertRaises(ValueError): plask.materialsdb.get("Al(0.2)GaAs:Si=1e14")

        print plask.materialsdb.all
        m = plask.materialsdb.get("Al(0.2)GaAs:Dp=3.0")
        self.assertEqual( m.__class__, Material.AlGaAsDp )
        self.assertEqual( m.name, "AlGaAs:Dp" )
        self.assertEqual( m.VBO(1.0), 3.0 )
        self.assertAlmostEqual( m.CBO(1.0), 0.8 )

        if sys.version > "2.7":
            with(self.assertRaisesRegexp(KeyError, 'N not allowed in material AlGaAs:Dp')): m = Material.AlGaAsDp(Al=0.2, N=0.9)

        AlGaAs = lambda **kwargs: plask.materialsdb.get("AlGaAs", **kwargs)
        m = AlGaAs(Al=0.2, dopant="Dp", dc=5.0)
        self.assertEqual( m.name, "AlGaAs:Dp" )
        self.assertEqual( m.VBO(), 1500.0 )
        correct = dict(Al=0.2, Ga=0.8, As=1.0)
        for k in correct:
            self.assertAlmostEqual( m.composition[k], correct[k] )
        del m
        self.assertEqual( ptest.materialName("Al(0.2)GaAs:Dp=3.0", plask.materialsdb), "AlGaAs:Dp" )
        self.assertEqual( ptest.materialVBO("Al(0.2)GaAs:Dp=3.0", plask.materialsdb, 1.0), 3.0 )

        c = Material.WithChar()
        self.assertEqual( c.name, "WithChar" )
        if sys.version >= 2.7:
            with self.assertRaisesRegexp(NotImplementedError, "Method not implemented"): c.VBO(1.0)
        self.assertEqual( ptest.call_chi(c, 'A'), 1.5)


    def testDefaultMaterials(self):
        self.assertIn( "GaN", plask.materialsdb )

    def testExistingMaterial(self):
        '''Test if existing materials works correctly'''
        m = plask.materialsdb.get("MyMaterial")
        self.assertEqual( plask.materialsdb.get("MyMaterial").name, "MyMaterial" )
        self.assertEqual( plask.materialsdb.get("MyMaterial").VBO(1.0), 0.5 )
        self.assertEqual( ptest.materialName("MyMaterial", plask.materialsdb), "MyMaterial" )
        self.assertEqual( ptest.materialVBO("MyMaterial", plask.materialsdb, 1.0), 0.5 )
        self.assertEqual( ptest.call_chi(m, 'B'), 1.0)
        self.assertEqual( m.VBO(), 150.0)
        self.assertEqual( m.chi(point='C'), 1.0)


    def testMaterialWithBase(self):
        mm = plask.materialsdb.get("MyMaterial")

        @plask.materials.simple
        class WithBase(plask.materials.Material):
            def __init__(self):
                super(WithBase, self).__init__(mm)

        m1 = WithBase()
        self.assertEqual( m1.name, "WithBase" )
        self.assertEqual( m1.VBO(1.0), 0.5 )

        m2 = plask.materialsdb.get("WithBase")
        self.assertEqual( m2.name, "WithBase" )
        self.assertEqual( m2.VBO(2.0), 1.0 )

        self.assertEqual( ptest.materialName("WithBase", plask.materialsdb), "WithBase" )
        self.assertEqual( ptest.materialVBO("WithBase", plask.materialsdb, 1.0), 0.5 )

    def testPassingMaterialsByName(self):
        mat = plask.geometry.Rectangle(2,2, "Al(0.2)GaAs:Dp=3.0").getMaterial(0,0)
        self.assertEqual( mat.name, "AlGaAs:Dp" )
        self.assertEqual( mat.VBO(1.0), 3.0 )

        if sys.version >= 2.7:
            with(self.assertRaises(ValueError)):
                plask.geometry.Rectangle(2,2, "Al(0.2)GaAs:Ja=3.0").getMaterial(0,0)
