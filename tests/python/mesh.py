#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

import plask, plask.mesh
import plasktest




class RectilinearMeshes(unittest.TestCase):

    def setUp(self):
        self.mesh2 = plask.mesh.Rectangular2D(plask.mesh.Rectilinear([1,3,2,1]), plask.mesh.Rectilinear(array([10,20], float)))
        self.mesh3 = plask.mesh.Rectangular3D(plask.mesh.Rectilinear([1,2,3]), plask.mesh.Rectilinear([10,20]), plask.mesh.Rectilinear([100,200]))

    def testOrdering2D(self):
        m = self.mesh2

        self.assertEqual( [list(i) for i in m], [[1,10], [1,20], [2,10], [2,20], [3,10], [3,20]] )
        self.assertEqual( m.minor_axis, m.axis1 )
        self.assertEqual( m.major_axis, m.axis0 )
        for i in range(len(m)):
            self.assertEqual( m.minor_index(i), m.index1(i) )
            self.assertEqual( m.major_index(i), m.index0(i) )

        m.ordering = '10'
        self.assertEqual( [list(i) for i in m], [[1,10], [2,10], [3,10], [1,20], [2,20], [3,20]] )

        m.ordering = 'best'
        self.assertEqual( [list(i) for i in m], [[1,10], [1,20], [2,10], [2,20], [3,10], [3,20]] )

        for i in range(len(m)):
            i0 = m.index0(i)
            i1 = m.index1(i)
            self.assertEqual( m.index(i0, i1), i )


    def testOrdering3D(self):
        m = self.mesh3

        self.assertEqual( [list(i) for i in m], [[1,10,100], [1,10,200], [1,20,100], [1,20,200],
                                                 [2,10,100], [2,10,200], [2,20,100], [2,20,200],
                                                 [3,10,100], [3,10,200], [3,20,100], [3,20,200]] )

        m.ordering = '210'
        self.assertEqual( [list(i) for i in m], [[1,10,100], [2,10,100], [3,10,100], [1,20,100], [2,20,100], [3,20,100],
                                                 [1,10,200], [2,10,200], [3,10,200], [1,20,200], [2,20,200], [3,20,200]] )
        m.ordering = '201'
        self.assertEqual( [list(i) for i in m], [[1,10,100], [1,20,100], [2,10,100], [2,20,100], [3,10,100], [3,20,100],
                                                 [1,10,200], [1,20,200], [2,10,200], [2,20,200], [3,10,200], [3,20,200]] )

        for i in range(len(m)):
            i0 = m.index0(i)
            i1 = m.index1(i)
            i2 = m.index2(i)
            self.assertEqual( m.index(i0, i1, i2), i )

        m.ordering = 'best'
        self.assertEqual( [list(i) for i in m], [[1,10,100], [1,10,200], [1,20,100], [1,20,200],
                                                 [2,10,100], [2,10,200], [2,20,100], [2,20,200],
                                                 [3,10,100], [3,10,200], [3,20,100], [3,20,200]] )
        self.assertEqual( m.minor_axis,  m.axis2 )
        self.assertEqual( m.medium_axis, m.axis1 )
        self.assertEqual( m.major_axis,  m.axis0 )
        for i in range(len(m)):
            self.assertEqual( m.minor_index(i),  m.index2(i) )
            self.assertEqual( m.middle_index(i), m.index1(i) )
            self.assertEqual( m.major_index(i),  m.index0(i) )
        plask.mesh.Rectangular2D(plask.mesh.Rectilinear([1,3,2,1]), plask.mesh.Rectilinear(array([10,20], float)))


    def testBoundary(self):
        self.mesh2.ordering = "10"
        geo = plask.geometry.Cartesian2D(plask.geometry.Rectangle(0,0,None))
        b = plask.mesh.Rectangular2D.Left()(self.mesh2, geo)
        self.assertIn(0, b)
        self.assertNotIn(1, b)
        self.assertIn(3, b)
        self.assertEqual( list(b), [0, 3] )

        self.assertEqual( plasktest.test_boundary(self.mesh2, geo, lambda mesh,i: i in [2,3]), [2, 3] )


    def testDivideGenerator(self):
        stack = plask.geometry.Stack2D()
        stack.append(plask.geometry.Rectangle(2, 2, None))
        stack.append(plask.geometry.Rectangle(2, 16, None))
        rect = plask.geometry.Rectangle(2, 16, None)
        stack.append(rect)

        generator1 = plask.mesh.Rectilinear2D.SimpleGenerator()
        generator2 = plask.mesh.Rectilinear2D.DivideGenerator()
        generator2.prediv = 2,2
        generator2.add_refinement("z", rect, 8.)

        self.assertEqual( list(generator2.get_refinements("z").values()), [[8.]] )

        mesh1 = generator1(stack)
        self.assertEqual( list(mesh1.axis0), [0., 2.] )
        self.assertEqual( list(mesh1.axis1), [0., 2., 18., 34.] )

        mesh2 = generator2(stack)
        self.assertEqual( list(mesh2.axis0), [0., 1., 2.] )
        self.assertEqual( list(mesh2.axis1), [0., 1., 2., 4., 6., 10., 18., 22., 26., 30., 34.] )

        generator3 = plask.mesh.Rectilinear2D.DivideGenerator()
        stack = plask.geometry.Stack2D()
        stack.append(plask.geometry.Rectangle(1000., 5., None))
        stack.append(plask.geometry.Rectangle(1000., 10., None))
        stack.append(plask.geometry.Rectangle(1000., 50., None))
        stack.append(plask.geometry.Rectangle(1000., 150., None))
        mesh3 = generator3(stack)
        self.assertEqual( list(mesh3.axis1), [0., 5., 15., 27.5, 40., 65.,102.5, 140., 215.] )

        generator1d = plask.mesh.Rectilinear1D.DivideGenerator()
        generator1d.postdiv = 2
        shelf = geometry.Shelf2D()
        shelf.append(geometry.Rectangle(1., 1., None))
        shelf.append(geometry.Rectangle(8., 1., None))
        self.assertEqual( list(generator1d(shelf)), [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0] )


    def testDivideGeneratorXML(self):
        manager = plask.Manager()
        manager.load('''
        <plask>
          <geometry>
            <cartesian2d name="main" axes="xy">
              <rectangle name="rect" dx="50" dy="5" material="GaAs"/>
            </cartesian2d>
          </geometry>
          <grids>
            <generator type="rectilinear2d" method="divide" name="refined">
              <refinements>
                <axis0 object="rect" by="5"/>
                <axis1 object="rect" every="1."/>
              </refinements>
            </generator>
            <generator type="rectilinear1d" method="divide" name="one">
              <refinements>
                <axis0 object="rect" by="2"/>
              </refinements>
            </generator>
          </grids>
        </plask>
        ''')
        msh = manager.meshgen['refined'](manager.geometry['rect'])
        self.assertEqual( list(msh.axis0), [0., 10., 20., 30., 40., 50.] )
        self.assertEqual( list(msh.axis1), [0., 1., 2., 3., 4., 5.] )
        self.assertEqual( list(manager.meshgen['one'](manager.geometry['rect'])), [0., 25., 50.] )


    def testRegenerationInSolver(self):
        stack = plask.geometry.Stack2D()
        rect = plask.geometry.Rectangle(2, 2, None)
        stack.append(rect)
        generator = plask.mesh.Rectilinear2D.DivideGenerator()

        solver = plasktest.SpaceTest()

        solver.mesh = generator
        self.assertFalse(solver.mesh_changed)

        solver.geometry = plask.geometry.Cartesian2D(stack)
        self.assertTrue(solver.mesh_changed)

        self.assertEqual( list(solver.mesh.axis1), [0., 2.] )

        solver.initialize()
        self.assertTrue(solver.initialized)
        self.assertFalse(solver.mesh_changed)

        generator.prediv[1] = 2
        self.assertTrue(solver.mesh_changed)
        self.assertEqual( list(solver.mesh.axis1), [0., 1., 2.] )
        generator.prediv = 3,3
        generator.prediv /= 3

        stack.append(rect)
        self.assertFalse(solver.initialized)
        self.assertEqual( list(solver.mesh.axis1), [0., 2., 4.] )


    def testEvents(self):
        test = plasktest.MeshTest()

        self.assertFalse(test.rectilinear2d_changed)
        test.rectilinear2d.axis0 = linspace(0., 10., 11)
        self.assertTrue(test.rectilinear2d_changed)
        test.rectilinear2d.axis0.insert(12)
        self.assertTrue(test.rectilinear2d_changed)
        test.rectilinear2d.axis0.extend(linspace(12., 15., 4))
        self.assertTrue(test.rectilinear2d_changed)
        test.rectilinear2d.axis1 = linspace(0., 10., 11)
        self.assertTrue(test.rectilinear2d_changed)
        test.rectilinear2d.axis1.insert(12)
        self.assertTrue(test.rectilinear2d_changed)
        test.rectilinear2d.axis1.extend(linspace(12., 15., 4))
        self.assertTrue(test.rectilinear2d_changed)

        self.assertFalse(test.rectilinear3d_changed)
        test.rectilinear3d.axis0 = linspace(0., 10., 11)
        self.assertTrue(test.rectilinear3d_changed)
        test.rectilinear3d.axis1 = linspace(0., 10., 11)
        self.assertTrue(test.rectilinear3d_changed)
        test.rectilinear3d.axis2 = linspace(0., 10., 11)
        self.assertTrue(test.rectilinear3d_changed)

        self.assertFalse(test.regular2d_changed)
        test.regular2d.axis0 = Regular(0., 10., 11)
        self.assertTrue(test.regular2d_changed)
        test.regular2d.axis0.start = 2.
        self.assertTrue(test.regular2d_changed)
        test.regular2d.axis0.stop = 12.
        self.assertTrue(test.regular2d_changed)
        test.regular2d.axis0.resize(20)
