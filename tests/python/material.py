#!/usr/bin/env plask
# -*- coding: utf-8 -*-
import unittest

from numpy import *
import sys

import plask
from plask import material, geometry
import plasktest as ptest

class Material(unittest.TestCase):

    @material.alloy('AlGaAs')
    class AlGaAs_fake(material.Material):
        def __init__(self, **kwargs):
            print(kwargs)
            super(Material.AlGaAs_fake, self).__init__(**kwargs)
            ptest.print_ptr(self)
        def __del__(self):
            ptest.print_ptr(self)
        def VB(self, T=300., e=0., point='G', hole='H'):
            return 2.*T
        def nr(self, lam, T=300., n=0.):
            """
            T range: 300:400
            source: Just imagination
            note: This is a test
            """
            return 3.5
        def absp(self, lam, T):
            return 0.

    @material.alloy()
    class AlGaAsDp(material.Material):
        name = "AlGaAs:Dp"
        def __init__(self, *args, **kwargs):
            super(Material.AlGaAsDp, self).__init__(**kwargs)
            print(kwargs)
            print("Composition: %s" % self.composition)
        def __del__(self):
            ptest.print_ptr(self)
        def VB(self, T=300., e=0., point='G', hole='H'):
            return self.doping * T
        def CB(self, T=300., e=0., point='G'):
            return self.composition['Ga'] * T
        def NR(self, lam, T, n):
            return (3.5, 3.6, 3.7, 0.1)

    @material.simple()
    class WithChar(material.Material):
        def chi(self, T, e, p):
            """
            T range: 300:400
            source: Simplicity
            """
            print("WithChar: %s" % p)
            return 1.5

    def setUp(self):
        ptest.add_my_material(material.db)

    def testCustomMaterial(self):
        '''Test creation of custom materials'''

        self.assertIn( "AlGaAs", material.db )
        self.assertIn( "AlGaAs_fake", material.db )

        m = Material.AlGaAs_fake(Al=0.1)
        self.assertEqual(m.name, "AlGaAs_fake")
        self.assertEqual( m.VB(1.0), 2.0 )
        self.assertEqual( m.nr(980., 300.), 3.5 )
        self.assertEqual( tuple(m.NR(980., 300.)), (3.5, 3.5, 3.5, 0.) )
        self.assertEqual( m.thermk(), material.AlGaAs(Al=0.1).thermk() )
        self.assertNotEqual( m.nr(980., 300.), material.AlGaAs(Al=0.1).nr(980., 300.) )
        self.assertEqual( m, material.get('AlGaAs_fake', Al=0.1) )
        del m

        self.assertEqual( ptest.material_name("Al(0.2)GaAs_fake", material.db), "AlGaAs_fake" )
        self.assertEqual( ptest.material_VB("Al(0.2)GaAs_fake", material.db, 1.0), 2.0 )

        print(list(material.db))
        with self.assertRaises(ValueError): material.get("Al(0.2)GaAs:Np=1e14")

        print(list(material.db))
        m1 = material.get("Al(0.2)GaAs:Dp=3.0")
        self.assertEqual( m1.__class__, Material.AlGaAsDp )
        self.assertEqual( m1.name, "AlGaAs:Dp" )
        self.assertEqual( m1.VB(1.0), 3.0 )
        self.assertAlmostEqual( m1.CB(1.0), 0.8 )
        print(ptest.NR(m1))
        self.assertEqual( tuple(ptest.NR(m1)), (3.5, 3.6, 3.7, 0.1) )

        with(self.assertRaisesRegexp(TypeError, "'N' not allowed in material AlGaAs:Dp")):
            mx = Material.AlGaAsDp(Al=0.2, N=0.9, doping=1e18)

        m2 = material.AlGaAs(Al=0.2, dopant="Dp", doping=5.0)
        self.assertEqual( m2.name, "AlGaAs:Dp" )
        self.assertEqual( m2.VB(), 1500.0 )
        correct = dict(Al=0.2, Ga=0.8, As=1.0)
        for k in correct:
            self.assertAlmostEqual( m2.composition[k], correct[k] )

        self.assertEqual( m1, material.get("Al(0.2)GaAs:Dp=3.0") )
        self.assertNotEqual( m1, m2 )

        self.assertEqual( ptest.material_name("Al(0.2)GaAs:Dp=3.0", material.db), "AlGaAs:Dp" )
        self.assertEqual( ptest.material_VB("Al(0.2)GaAs:Dp=3.0", material.db, 1.0), 3.0 )

        c = Material.WithChar()
        self.assertEqual( c.name, "WithChar" )
        with self.assertRaisesRegexp(NotImplementedError, "Method not implemented"): c.VB(1.0)
        self.assertEqual( ptest.call_chi(c, 'A'), 1.5)

    def testInfo(self):
        self.assertEqual( material.info("WithChar"), {'chi': {'source': "Simplicity", 'ranges': {'T': (300., 400.)}}} )
        self.assertEqual( material.info("AlGaAs_fake"), {'nr': {'source': "Just imagination",
                                                                'ranges': {'T': (300., 400.)},
                                                                'note': "This is a test"}} )

    def testDefaultMaterials(self):
        self.assertIn( "GaN", material.db )
        self.assertEqual( str(material.AlGaN(Al=0.2)), "Al(0.2)GaN" )
        self.assertRegex( str(material.AlGaN(Ga=0.8, dopant="Si", doping=1e17)), r"Al\(0\.2\)GaN:Si=1e\+0?17" )

    def testExistingMaterial(self):
        '''Test if existing materials works correctly'''
        m = material.get("MyMaterial")
        self.assertEqual( material.get("MyMaterial").name, "MyMaterial" )
        self.assertEqual( material.get("MyMaterial").VB(1.0), 0.5 )
        self.assertEqual( ptest.material_name("MyMaterial", material.db), "MyMaterial" )
        self.assertEqual( ptest.material_VB("MyMaterial", material.db, 1.0), 0.5 )
        self.assertEqual( ptest.call_chi(m, 'B'), 1.0 )
        self.assertEqual( m.VB(), 150.0 )
        self.assertEqual( m.chi(point='C'), 1.0 )


    def testMaterialWithBase(self):
        mm = material.get("MyMaterial")

        @material.simple(mm)
        class WithBase(material.Material):
            def __init__(self, **kwargs):
                super(WithBase, self).__init__()

        m1 = WithBase()
        self.assertEqual( m1.name, "WithBase" )
        self.assertEqual( m1.VB(1.0), 0.5 )

        m2 = material.get("WithBase")
        self.assertEqual( m2.name, "WithBase" )
        self.assertEqual( m2.VB(2.0), 1.0 )

        self.assertEqual( ptest.material_name("WithBase", material.db), "WithBase" )
        self.assertEqual( ptest.material_VB("WithBase", material.db, 1.0), 0.5 )


    def testMaterialWithBaseAndDoping(self):
        @material.simple('Al(0.2)GaAs:Si')
        class WithBase2(material.Material):
            name="WithBase2:Si"
            def __init__(self):
                super(WithBase2, self).__init__(doping=1e18)

        m1 = WithBase2()
        m2 = material.get('Al(0.2)GaAs:Si=1e18')
        self.assertEqual( m1.cond(), m2.cond() )

    def testAlloyMaterialWithBaseAndDoping(self):
        @material.simple('AlGaAs:Si=1e18')
        class AlGaAs_1_Si(material.Material):
            name="AlGaAs_1:Si"
            def __init__(self):
                super(AlGaAs_1_Si, self).__init__(Al=0.2, doping=1e5)

        m1 = AlGaAs_1_Si()
        m2 = material.get('Al(0.2)GaAs:Si=1e18')
        self.assertEqual( m1.doping, 1e5 )
        self.assertEqual( m1.cond(), m2.cond() )

    def testAlloyMaterialWithUndopedBaseAndDoping(self):
        @material.alloy('AlGaAs')
        class AlGaAs_2_Si(material.Material):
            name="AlGaAs_2:Si"
            def __init__(self):
                super(AlGaAs_2_Si, self).__init__(Al=0.2, doping=1e18)

        m = AlGaAs_2_Si()
        self.assertEqual( m.doping, 1e18 )

    def testPassingMaterialsByName(self):
        mat = plask.geometry.Rectangle(2,2, "Al(0.2)GaAs:Dp=3.0").get_material(0,0)
        self.assertEqual( mat.name, "AlGaAs:Dp" )
        self.assertEqual( mat.VB(1.0), 3.0 )

        with(self.assertRaises(ValueError)): plask.geometry.Rectangle(2,2, "Al(0.2)GaAs:Ja=3.0").get_material(0,0)


    def testThermK(self):
        @material.simple()
        class Therm(material.Material):
            def thermk(self, T, t): return T + t

        self.assertEqual( ptest.material_thermk("Therm", material.db, 300.), (infty,infty) )
        self.assertEqual( ptest.material_thermk("Therm", material.db, 300., 2.), (302.,302.) )


    def testComparison(self):
        @material.simple('GaAs')
        class Mat1(material.Material):
            def __init__(self, val):
                super(Mat1, self).__init__()
                self.val = val
        @material.simple('AlAs')
        class Mat2(material.Material):
            def __init__(self, val):
                super(Mat2, self).__init__()
                self.val = val
        m1 = Mat1(1)
        m2 = Mat1(2)
        m3 = Mat2(1)
        m4 = Mat1(1)
        self.assertNotEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertEqual(m1, m4)

        @material.simple()
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

    def testSelfBase(self):
        plask.loadxpl('''
          <plask>
            <materials>
              <material name="bas" base="semiconductor">
                <thermk>45.</thermk>
              </material>
              <material name="xmat" base="bas">
                <A>2. * self.B()</A>
                <B>10.</B>
                <thermk>2. * array(self.base.thermk())</thermk>
              </material>
            </materials>
          </plask>
        ''')
        xmat = material.get('xmat')
        self.assertEqual( xmat.A(), 20. )
        self.assertEqual( xmat.thermk(), (90., 90.) )

        @material.simple('bas')
        class Mat(material.Material):
            def thermk(self, T=300., h=infty):
                val = 3. * self.base.thermk(T, h)[0]
                return (val, val)
        mat = material.get('Mat')
        self.assertEqual( Mat().thermk(), (135., 135.) )
        self.assertEqual( mat.thermk(), (135., 135.) )


    def testXmlAlloy(self):
        cond = material.get('Al(0.2)GaAs:Si=1e18').cond()
        plask.loadxpl('''
          <plask>
            <materials>
              <material name="AlGaAs_my" base="GaAs" alloy="yes">
                <thermk>10*self.Ga</thermk>
              </material>
              <material name="AlGaAs:Si" base="AlGaAs:Si" alloy="yes">
                <thermk>2*self.Al</thermk>
              </material>
            </materials>
          </plask>
        ''')
        self.assertEqual( material.get('Al(0.2)GaAs_my').thermk(), (8.0, 8.0) )
        algas = material.get('Al(0.2)GaAs:Si=1e18')
        self.assertEqual( str(algas), 'Al(0.2)GaAs:Si=1e+18' )
        self.assertEqual( algas.thermk(), (0.4, 0.4) )
        self.assertEqual( algas.cond(), cond )
        algas0 = material.get('Al(0.2)GaAs:Si=1e18')
        algas1 = material.get('Al(0.3)GaAs:Si=1e18')
        self.assertEqual( algas, algas0 )
        self.assertNotEqual( algas, algas1 )

    def testGradientMaterial(self):
        rect = geometry.Rectangle(2., 2., ('Al(0.2)GaAs', 'Al(0.8)GaAs'))
        m1 = rect.get_material(1., 0.000001)
        m2 = rect.representative_material
        m3 = rect.get_material(1., 1.999999)
        self.assertAlmostEqual( m1.composition['Al'], 0.2, 4 )
        self.assertAlmostEqual( m2.composition['Al'], 0.5, 4 )
        self.assertAlmostEqual( m3.composition['Al'], 0.8, 4 )
        self.assertEqual( rect.material, (material.get('Al(0.2)GaAs'), material.get('Al(0.8)GaAs')) )

    def testCustomProvider(self):
        @material.simple('GaAs')
        class ChangingMaterial(material.Material):
            def __init__(self, point):
                super(ChangingMaterial, self).__init__()
                self.point = point
            def cond(self, T=300.):
                return self.point[0]**2
        def get_material(p):
            return ChangingMaterial(p)
        rect = geometry.Rectangle(4., 2., get_material)
        m1 = rect.get_material(1., 1.)
        m2 = rect.representative_material
        m3 = rect.get_material(3., 1.)
        self.assertAlmostEqual( m1.cond(), 0.25**2, 4 )
        self.assertAlmostEqual( m2.cond(), 0.50**2, 4 )
        self.assertAlmostEqual( m3.cond(), 0.75**2, 4 )
        self.assertIs( rect.material, get_material )

    def testTemporaryDB(self):
        @material.simple('semiconductor')
        class Test(material.Material):
            def A(self, T=300.):
                return 1
        with material.savedb() as saved:
            @material.simple('semiconductor')
            class Test(material.Material):
                def A(self, T=300.):
                    return 2
            self.assertEqual( material.get('Test').A(), 2 )
            self.assertEqual( saved.get('Test').A(), 1 )
        self.assertEqual( material.get('Test').A(), 1 )

    def testMaterialWithParams(self):
        m1 = material.get('[nr=2.5 absp=2e5]')
        m2 = material.with_params(nr=2.5, absp=2e5)
        self.assertAlmostEqual( m1.Nr(980), 2.5-1.556j, 2 )
        self.assertEqual( str(m1), '[absp=200000 nr=2.5]' )
        self.assertEqual( m1, m2 )

    def testMaterialWithParamsAndBase(self):
        m0 = material.get('GaAs')
        m1 = material.get('GaAs [nr=2.5 absp=2e5]')
        m2 = material.with_params('GaAs', nr=2.5, absp=2e5)
        self.assertEqual( m1.cond(), m0.cond() )
        self.assertAlmostEqual( m1.Nr(980), 2.5-1.556j, 2 )
        self.assertEqual( str(m1), 'GaAs [absp=200000 nr=2.5]' )
        self.assertEqual( str(m2), 'GaAs [absp=200000 nr=2.5]' )
        self.assertEqual( m1, m2 )


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
