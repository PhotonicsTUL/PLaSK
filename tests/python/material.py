#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *
import sys

import plask.material
import plasktest as ptest


class Material(unittest.TestCase):

    @plask.material.complex
    class AlGaAs(plask.material.Material):
        def __init__(self, **kwargs):
            super(Material.AlGaAs, self).__init__()
            ptest.print_ptr(self)
        def __del__(self):
            ptest.print_ptr(self)
        def VBO(self, T=300.):
            return 2.*T

    @plask.material.complex
    class AlGaAsDp(plask.material.Material):
        name = "AlGaAs:Dp"
        def __init__(self, *args, **kwargs):
            super(Material.AlGaAsDp, self).__init__()
            self.args = args,
            print(kwargs)
            self.kwargs = kwargs
            self.composition = self._completeComposition(kwargs, self.name);
            print(self.composition)
        def __del__(self):
            ptest.print_ptr(self)
        def VBO(self, T=300.):
            return self.kwargs['dc'] * T
        def CBO(self, T=300.):
            return self.composition['Ga'] * T

    @plask.material.simple
    class WithChar(plask.material.Material):
        def chi(self, T, p):
            print("WithChar: %s" % p)
            return 1.5

    def setUp(self):
        ptest.addMyMaterial(plask.materialdb)


    def testMaterial(self):
        '''Test basic behavior of Material class'''
        result = plask.material.Material._completeComposition(dict(Ga=0.3, Al=None, As=0.0, N=None))
        correct = dict(Ga=0.3, Al=0.7, As=0.0, N=1.0)
        for k in correct:
            self.assertAlmostEqual( result[k], correct[k] )


    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        self.assertIn( "AlGaAs", plask.materialdb )

        m = Material.AlGaAs()
        self.assertEqual(m.name, "AlGaAs")
        self.assertEqual( m.VBO(1.0), 2.0 )
        del m

        self.assertEqual( ptest.materialName("Al(0.2)GaAs", plask.materialdb), "AlGaAs" )
        self.assertEqual( ptest.materialVBO("Al(0.2)GaAs", plask.materialdb, 1.0), 2.0 )

        print(plask.materialdb.all)
        with self.assertRaises(ValueError): plask.materialdb.get("Al(0.2)GaAs:Si=1e14")

        print(plask.materialdb.all)
        m = plask.materialdb.get("Al(0.2)GaAs:Dp=3.0")
        self.assertEqual( m.__class__, Material.AlGaAsDp )
        self.assertEqual( m.name, "AlGaAs:Dp" )
        self.assertEqual( m.VBO(1.0), 3.0 )
        self.assertAlmostEqual( m.CBO(1.0), 0.8 )

        with(self.assertRaisesRegexp(KeyError, 'N not allowed in material AlGaAs:Dp')): m = Material.AlGaAsDp(Al=0.2, N=0.9)

        AlGaAs = lambda **kwargs: plask.materialdb.get("AlGaAs", **kwargs)
        m = AlGaAs(Al=0.2, dopant="Dp", dc=5.0)
        self.assertEqual( m.name, "AlGaAs:Dp" )
        self.assertEqual( m.VBO(), 1500.0 )
        correct = dict(Al=0.2, Ga=0.8, As=1.0)
        for k in correct:
            self.assertAlmostEqual( m.composition[k], correct[k] )
        del m
        self.assertEqual( ptest.materialName("Al(0.2)GaAs:Dp=3.0", plask.materialdb), "AlGaAs:Dp" )
        self.assertEqual( ptest.materialVBO("Al(0.2)GaAs:Dp=3.0", plask.materialdb, 1.0), 3.0 )

        c = Material.WithChar()
        self.assertEqual( c.name, "WithChar" )
        with self.assertRaisesRegexp(NotImplementedError, "Method not implemented"): c.VBO(1.0)
        self.assertEqual( ptest.call_chi(c, 'A'), 1.5)


    def testDefaultMaterials(self):
        self.assertIn( "GaN", plask.materialdb )
        self.assertEqual( str(plask.material.AlGaN(Al=0.2)), "Al(0.2)GaN" )
        self.assertRegexpMatches( str(plask.material.AlGaN(Ga=0.8, dopant="Si", dc=1e17)), r"Al\(0\.2\)GaN:Si=1e\+0?17" )
        self.assertEqual( ptest.materialTypeId(plask.material.Material("GaN")), ptest.materialTypeId(plask.materialdb.get("GaN")) )

    def testExistingMaterial(self):
        '''Test if existing materials works correctly'''
        m = plask.materialdb.get("MyMaterial")
        self.assertEqual( plask.materialdb.get("MyMaterial").name, "MyMaterial" )
        self.assertEqual( plask.materialdb.get("MyMaterial").VBO(1.0), 0.5 )
        self.assertEqual( ptest.materialName("MyMaterial", plask.materialdb), "MyMaterial" )
        self.assertEqual( ptest.materialVBO("MyMaterial", plask.materialdb, 1.0), 0.5 )
        self.assertEqual( ptest.call_chi(m, 'B'), 1.0)
        self.assertEqual( m.VBO(), 150.0)
        self.assertEqual( m.chi(point='C'), 1.0)


    def testMaterialWithBase(self):
        mm = plask.materialdb.get("MyMaterial")

        @plask.material.simple
        class WithBase(plask.material.Material):
            def __init__(self):
                super(WithBase, self).__init__(mm)

        m1 = WithBase()
        self.assertEqual( m1.name, "WithBase" )
        self.assertEqual( m1.VBO(1.0), 0.5 )

        m2 = plask.materialdb.get("WithBase")
        self.assertEqual( m2.name, "WithBase" )
        self.assertEqual( m2.VBO(2.0), 1.0 )

        self.assertEqual( ptest.materialName("WithBase", plask.materialdb), "WithBase" )
        self.assertEqual( ptest.materialVBO("WithBase", plask.materialdb, 1.0), 0.5 )


    def testPassingMaterialsByName(self):
        mat = plask.geometry.Rectangle(2,2, "Al(0.2)GaAs:Dp=3.0").getMaterial(0,0)
        self.assertEqual( mat.name, "AlGaAs:Dp" )
        self.assertEqual( mat.VBO(1.0), 3.0 )

        with(self.assertRaises(ValueError)): plask.geometry.Rectangle(2,2, "Al(0.2)GaAs:Ja=3.0").getMaterial(0,0)
