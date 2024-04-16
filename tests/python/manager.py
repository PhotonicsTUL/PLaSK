#!/usr/bin/env plask
# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import unittest
import sys

import plask
from plask import material


class ManagerTest(unittest.TestCase):

    def setUp(self):
        self.manager = plask.Manager()
        self.manager.load('''
        <plask>
          <defines>
            <define name="y" value="2"/>
          </defines>
          <geometry>
            <cartesian2d name="Space-1" axes="xy">
              <stack name="Stack-2">
                <item path="Path-4"><rectangle name="Block-3" dx="5" dy="{y}" material="GaN" /></item>
                <again ref="Block-3"/>
              </stack>
            </cartesian2d>
          </geometry>
          <grids>
            <mesh type="rectangular2d" name="lin">
              <axis0>1, 2, 3</axis0>
              <axis1 type="ordered" start="10" stop="30" num="3"/>
            </mesh>
            <mesh type="rectangular2d" name="reg">
              <axis0 start="10" stop="30" num="3"/>
              <axis1 start="1" stop="3" num="3"/>
            </mesh>
            <mesh type="rectangular2d" name="geo">
              <axis0 start="0" stop="{self.geo.Block_3.width}" num="3"/>
              <axis1 start="0" stop="{GEO.Block_3.height}" num="1"/>
            </mesh>
            <generator type="rectangular2d" method="divide" name="test">
              <prediv by="4"/>
              <postdiv by0="2" by1="3"/>
            </generator>
            <generator type="rectangular2d" method="divide" name="refined">
              <postdiv by0="2"/>
              <refinements>
                <axis0 object="Block-3" at="1.0"/>
                <axis1 object="Block-3" path="Path-4" at="1.0"/>
              </refinements>
            </generator>
          </grids>
        </plask>
        ''')

    def testRefcount(self):
        self.assertEqual(sys.getrefcount(self.manager), 2)

    def testGeometry(self):
        self.assertEqual(len(self.manager.geo), 3)
        self.assertEqual(type(self.manager.geo["Block-3"]), plask.geometry.Block2D)
        self.assertEqual(list(self.manager.geo["Stack-2"].get_leafs_bboxes()),
            [plask.geometry.Box2D(0,0,5,2), plask.geometry.Box2D(0,2,5,4)])
        self.assertEqual(type(self.manager.geo.Space_1), plask.geometry.Cartesian2D)
        self.assertEqual(len(self.manager.pth), 1)
        with self.assertRaises(KeyError): self.manager.geo["nonexistent"]

    def testDictionaries(self):
        self.assertEqual(list(self.manager.geo), ["Block_3", "Space_1", "Stack_2"])

    def testExport(self):
        self.manager.export(globals())
        self.assertIn("Space-1", GEO)
        self.assertEqual(type(GEO.Space_1), plask.geometry.Cartesian2D)

    def testMesh(self):
        self.assertEqual(len(self.manager.msh), 5)
        self.assertEqual(self.manager.msh.lin.axis0 , [1, 2, 3])
        self.assertEqual(self.manager.msh.lin.axis1 , [10, 20, 30])
        self.assertEqual(list(self.manager.msh["reg"].axis1) , [1, 2, 3])
        self.assertEqual(list(self.manager.msh["reg"].axis0) , [10, 20, 30])

    def testGenerators(self):
        self.assertEqual(tuple(self.manager.msh.test.prediv), (4,4))
        self.assertEqual(tuple(self.manager.msh.test.postdiv), (2,3))

        mesh = self.manager.msh.refined.generate(self.manager.geo.Stack_2)
        self.assertEqual(mesh.axis1, [0., 2., 3., 4.])
        self.assertEqual(mesh.axis0, [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])

    def testException(self):
        manager = plask.Manager()
        with self.assertRaises(plask.XMLError):
            manager.load('''
            <plask>
              <geometry>
                <cartesian2d name="Space-1" axes="xy">
                  <stack name="Stack-2">
                    <item path="Path-4"><rectangle name="Block-3" x="5" y="2" material="GaN" /></item>
                    <again ref="Block-3"/>
                </cartesian2d>
              </geometry>
            </plask>
            ''')
        with self.assertRaises(plask.XMLError):
            manager.load('''
              <plask>
                <grids>
                  <mesh type="rectangular2d" name="lin">
                    <axis0>1, 2, 3</axis0>
                    <axis0>10 20 30</axis0>
                  </mesh>
                </grids>
              <plask>
            ''')
        with self.assertRaises(plask.XMLError):
            manager.load('''
            <plask>
              <grids>
                <generator type="rectangular2d" method="divide" name="test">
                  <postdiv by="4" by0="2" by1="3"/>
                </generator>
              </grids>
            </plask>
            ''')
        with self.assertRaises(plask.XMLError):
            manager.load('''
            <plask>
              <grids>
                <generator type="rectangular2d" method="divide" name="test">
                  <postdiv bye="4"/>
                </generator>
              </grids>
            </plask>
            ''')
        with self.assertRaises(plask.XMLError):
            manager.load('''
            <plask>
              <geometry>
                <cartesian2d name="Space-2" axes="xy">
                  <rectangle x="5" y="2" material="GaN" />
                </cartesian2d>
            </plask>
            ''')

    def testSolverConnections(self):
        manager = plask.Manager()
        manager.load('''
        <plask>
          <solvers>
            <plasktest lib="solvers" solver="InOut" name="output"/>
            <plasktest lib="solvers" solver="InOut" name="input"/>
          </solvers>
          <connects>
            <connect out="output.outWavelength" in="input.inWavelength"/>
          </connects>
        </plask>
        ''')
        self.assertEqual(manager.solvers.output.inWavelength(0), 2)
        self.assertEqual(manager.solvers.input.inWavelength(0), 5)

    def testMaterials(self):
        manager = plask.Manager()
        manager.load('''
        <plask>
          <materials>
            <material name="XmlMat" base="dielectric">
              <nr>1. + 0.001*T + 0.0001*wl</nr>
              <absp>1.</absp>
            </material>
            <material name="XmlMat:Mg" base="GaN:Mg">
              <nr>1. + 0.001*T + 0.0001*wl</nr>
              <absp>T * self.doping</absp>
            </material>
            <material name="XmlMatMg20" base="GaN:Mg=1e20">
              <nr>1. + 0.001*T + 0.0001*wl</nr>
              <absp>T * self.doping</absp>
            </material>
            <material name="XmlMatSimple" base="dielectric">
              <nr>3.5</nr>
              <absp>0.</absp>
            </material>
          </materials>
        </plask>
        ''')
        material.update_factories()
        mat = plask.material.XmlMat()
        self.assertAlmostEqual(mat.nr(900, 300), 1.39)
        self.assertAlmostEqual(mat.Nr(900, 300), 1.39-7.16197244e-06j)
        self.assertEqual(tuple(plask.material.XmlMatSimple().Eps(900, 300)), (12.25, 12.25, 12.25, 0., 0., 0., 0., 0., 0.))


        mad = plask.material.XmlMat(dopant="Mg", doping=1e18)
        self.assertEqual(mad.cond(300), material.GaN(dopant="Mg", doping=1e18).cond(300))
        self.assertEqual(mad.absp(900, 300), 300 * 1e18)

        mad20 = plask.material.XmlMatMg20()
        self.assertEqual(mad20.cond(300), material.GaN(dopant="Mg", doping=1e20).cond(300))

    def testVariables(self):
        manager = plask.Manager()
        manager.load('''
        <plask>
          <defines>
            <define name="hh1" value="9"/>
            <define name="h2" value="1"/>
            <define name="mat" value="'Al'"/>
          </defines>
          <geometry>
            <cartesian2d axes="xy">
              <stack>
                <rectangle name="block1" dx="5" dy="{sqrt(hh1)}" material="{mat}{'As'}"/>
                <rectangle name="block2" dx="{self.geo.block1.dims[0]}" dy="{h2}" material="GaAs"/>
              </stack>
            </cartesian2d>
          </geometry>
        </plask>
        ''', {'hh1': 4})
        self.assertEqual(str(manager.geo.block1.material), 'AlAs')
        self.assertEqual(manager.geo.block1.dims[1], 2)
        self.assertEqual(manager.geo.block2.dims[0], 5)
        self.assertEqual(manager.geo.block2.dims[1], 1)


class FakeModule:

    class CustomSolver(Solver):
        def load_xpl(self, xpl, manager):
            for tag in xpl:
                if tag == 'something':
                    for subtag in tag:
                        if subtag == 'withtext':
                            for item in subtag:
                                self.text = item.text
                elif tag == 'config':
                    self.attrs = tag.attrs
                    self.a = 2 * tag['a']
                    self.b = tag.get('b', 0)
                    self.c = tag['c']
                elif tag == 'geometry':
                    self.geometry = manager.geo[tag['ref']]

sys.modules['fake'] = FakeModule


class CustomSolverTest(unittest.TestCase):

    def testCustomSolver(self):
        manager = plask.Manager()
        manager.load('''
        <plask>
          <defines>
            <define name="x" value="2"/>
          </defines>
          <geometry>
            <cartesian2d name="main" axes="xy">
              <stack>
                <rectangle dx="5" dy="2" material="GaN" />
              </stack>
            </cartesian2d>
          </geometry>
          <solvers>
            <local name="custom" solver="CustomSolver" lib="fake">
              <geometry ref="main"/>
              <something>
                <withtext>
                  <val>passed</val>
                </withtext>
              </something>
              <config a="{x}" c="ok" d="yes"/>
            </local>
          </solvers>
        </plask>
        ''')
        solver = manager.solvers['custom']
        self.assertIsInstance(solver, FakeModule.CustomSolver)
        self.assertEqual(solver.text, "passed")
        self.assertEqual(solver.a, 4)
        self.assertEqual(solver.b, 0)
        self.assertEqual(solver.c, "ok")
        self.assertEqual(solver.attrs, dict(a=2, c='ok', d=True))
        self.assertEqual(solver.geometry, manager.geo.main)


@plask.Manager._geometry_changer('test')
class TestChanger:
    def __init__(self, xml, manager):
        self.delete = xml['delete']

    def __call__(self, obj):
        if isinstance(obj, (plask.geometry.GeometryObjectLeaf2D, plask.geometry.GeometryObjectLeaf3D)):
            roles = list(obj.roles)
            if self.delete in roles:
                return []
            elif roles:
                obj.material = roles[-1]
                return obj


class XplChangerTest(unittest.TestCase):

    def setUp(self):
        self.manager = plask.Manager()
        self.manager.load('''
        <plask>
          <geometry>
            <cartesian2d name="test1" axes="x,y">
              <stack>
                <rectangle material="Cu" dx="2" dy="1" name="Cu"/>
                <rectangle material="GaAs" dx="2" dy="1" role="GaN"/>
                <rectangle material="AlAs" dx="2" dy="1" role="del"/>
                <rectangle material="InAs" dx="2" dy="1"/>
              </stack>
            </cartesian2d>
            <cartesian2d name="test2" axes="x,y">
              <copy from="test1">
                <test delete='del'/>
                <delete object="Cu"/>
              </copy>
            </cartesian2d>
            <cartesian2d name="test3" axes="x,y">
              <rectangle material="GaAs" dx="1" dy="1"/>
            </cartesian2d>
          </geometry>
        </plask>
        ''')

    def testChanger(self):
        self.assertEqual(str(self.manager.geo.test2.get_material(1, 0.5)), 'InAs')
        self.assertEqual(str(self.manager.geo.test2.get_material(1, 1.5)), 'GaN')
        self.assertEqual(str(self.manager.geo.test2.get_material(1, 2.5)), 'air')
        self.assertEqual(str(self.manager.geo.test2.get_material(1, 3.5)), 'air')


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
