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
        def VB(self, T=300., e=0., point='G', hole='H'):
            return 2.*T
        def nr(self, wl, T):
            return 3.5
        def absp(self, wl, T):
            return 0.

    @plask.material.complex
    class AlGaAsDp(plask.material.Material):
        name = "AlGaAs:Dp"
        def __init__(self, *args, **kwargs):
            super(Material.AlGaAsDp, self).__init__()
            self.args = args,
            print(kwargs)
            self.kwargs = kwargs
            self.composition = self.complete_composition(kwargs, self.name);
            print(self.composition)
        def __del__(self):
            ptest.print_ptr(self)
        def VB(self, T=300., e=0., point='G', hole='H'):
            return self.kwargs['dc'] * T
        def CB(self, T=300., e=0., point='G'):
            return self.composition['Ga'] * T
        def NR(self, wl, T):
            return (3.5, 3.6, 3.7, 0.1, 0.2)

    @plask.material.simple
    class WithChar(plask.material.Material):
        def chi(self, T, e, p):
            print("WithChar: %s" % p)
            return 1.5

    def setUp(self):
        ptest.add_my_material(plask.materialdb)

    def testMaterial(self):
        '''Test basic behavior of Material class'''
        result = plask.material.Material.complete_composition(dict(Ga=0.3, Al=None, As=0.0, N=None))
        correct = dict(Ga=0.3, Al=0.7, As=0.0, N=1.0)
        for k in correct:
            self.assertAlmostEqual( result[k], correct[k] )

    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        self.assertIn( "AlGaAs", plask.materialdb )

        m = Material.AlGaAs()
        self.assertEqual(m.name, "AlGaAs")
        self.assertEqual( m.VB(1.0), 2.0 )
        self.assertEqual( m.nr(980., 300.), 3.5 )
        self.assertEqual( m.NR(980., 300.), (3.5, 3.5, 3.5, 0., 0.) )
        del m

        self.assertEqual( ptest.material_name("Al(0.2)GaAs", plask.materialdb), "AlGaAs" )
        self.assertEqual( ptest.material_VB("Al(0.2)GaAs", plask.materialdb, 1.0), 2.0 )

        print(plask.materialdb.all)
        with self.assertRaises(ValueError): plask.materialdb.get("Al(0.2)GaAs:Np=1e14")

        print(plask.materialdb.all)
        m = plask.materialdb.get("Al(0.2)GaAs:Dp=3.0")
        self.assertEqual( m.__class__, Material.AlGaAsDp )
        self.assertEqual( m.name, "AlGaAs:Dp" )
        self.assertEqual( m.VB(1.0), 3.0 )
        self.assertAlmostEqual( m.CB(1.0), 0.8 )
        self.assertEqual( ptest.NR(m), (3.5, 3.6, 3.7, 0.1, 0.2) )

        with(self.assertRaisesRegexp(TypeError, "'N' not allowed in material AlGaAs:Dp")): m = Material.AlGaAsDp(Al=0.2, N=0.9)

        AlGaAs = lambda **kwargs: plask.materialdb.get("AlGaAs", **kwargs)
        m = AlGaAs(Al=0.2, dp="Dp", dc=5.0)
        self.assertEqual( m.name, "AlGaAs:Dp" )
        self.assertEqual( m.VB(), 1500.0 )
        correct = dict(Al=0.2, Ga=0.8, As=1.0)
        for k in correct:
            self.assertAlmostEqual( m.composition[k], correct[k] )
        del m
        self.assertEqual( ptest.material_name("Al(0.2)GaAs:Dp=3.0", plask.materialdb), "AlGaAs:Dp" )
        self.assertEqual( ptest.material_VB("Al(0.2)GaAs:Dp=3.0", plask.materialdb, 1.0), 3.0 )

        c = Material.WithChar()
        self.assertEqual( c.name, "WithChar" )
        with self.assertRaisesRegexp(NotImplementedError, "Method not implemented"): c.VB(1.0)
        self.assertEqual( ptest.call_chi(c, 'A'), 1.5)


    def testDefaultMaterials(self):
        self.assertIn( "GaN", plask.materialdb )
        self.assertEqual( str(plask.material.AlGaN(Al=0.2)), "Al(0.2)GaN" )
        self.assertRegexpMatches( str(plask.material.AlGaN(Ga=0.8, dp="Si", dc=1e17)), r"Al\(0\.2\)GaN:Si=1e\+0?17" )

    def testExistingMaterial(self):
        '''Test if existing materials works correctly'''
        m = plask.materialdb.get("MyMaterial")
        self.assertEqual( plask.materialdb.get("MyMaterial").name, "MyMaterial" )
        self.assertEqual( plask.materialdb.get("MyMaterial").VB(1.0), 0.5 )
        self.assertEqual( ptest.material_name("MyMaterial", plask.materialdb), "MyMaterial" )
        self.assertEqual( ptest.material_VB("MyMaterial", plask.materialdb, 1.0), 0.5 )
        self.assertEqual( ptest.call_chi(m, 'B'), 1.0)
        self.assertEqual( m.VB(), 150.0)
        self.assertEqual( m.chi(point='C'), 1.0)


    def testMaterialWithBase(self):
        mm = plask.materialdb.get("MyMaterial")

        @plask.material.simple
        class WithBase(plask.material.Material):
            def __init__(self):
                super(WithBase, self).__init__(mm)

        m1 = WithBase()
        self.assertEqual( m1.name, "WithBase" )
        self.assertEqual( m1.VB(1.0), 0.5 )

        m2 = plask.materialdb.get("WithBase")
        self.assertEqual( m2.name, "WithBase" )
        self.assertEqual( m2.VB(2.0), 1.0 )

        self.assertEqual( ptest.material_name("WithBase", plask.materialdb), "WithBase" )
        self.assertEqual( ptest.material_VB("WithBase", plask.materialdb, 1.0), 0.5 )


    def testPassingMaterialsByName(self):
        mat = plask.geometry.Rectangle(2,2, "Al(0.2)GaAs:Dp=3.0").get_material(0,0)
        self.assertEqual( mat.name, "AlGaAs:Dp" )
        self.assertEqual( mat.VB(1.0), 3.0 )

        with(self.assertRaises(ValueError)): plask.geometry.Rectangle(2,2, "Al(0.2)GaAs:Ja=3.0").get_material(0,0)


    def testThermK(self):
        @material.simple
        class Therm(material.Material):
            def thermk(self, T, t): return T + t

        self.assertEqual( ptest.material_thermk("Therm", materialdb, 300.), (infty,infty) )
        self.assertEqual( ptest.material_thermk("Therm", materialdb, 300., 2.), (302.,302.) )


    def testComparison(self):

        @material.simple
        class Mat(material.Material):
            def __init__(self, base, val):
                super(Mat, self).__init__(base)
                self.val = val
        m1 = Mat("GaAs", 1)
        m2 = Mat("GaAs", 2)
        m3 = Mat("AlAs", 1)
        m4 = Mat("GaAs", 1)
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertEqual(m1, m4)

        @material.simple
        class Nat(material.Material):
            def __init__(self, val):
                super(Nat, self).__init__()
                self.val = val
            def __eq__(self, other):
                return self.val != other.val
        n1 = Nat(1)
        n2 = Nat(2)
        n3 = Nat(1)
        self.assertTrue(ptest.compareMaterials(n1, n2))
        self.assertFalse(ptest.compareMaterials(n1, n3))
