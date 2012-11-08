#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import unittest

import plask.mesh
import plasktest


class ReceiverTest(unittest.TestCase):

    def setUp(self):
        self.solver = plasktest.SimpleSolver()
        self.mesh1 = plask.mesh.Regular2D((0., 4., 3), (0., 20., 3))
        self.mesh2 = self.mesh1.get_midpoints();


    def testReceiverWithConstant(self):
        self.solver.inTemperature = 250
        self.assertEqual( list(self.solver.inTemperature(self.mesh2)), [250., 250., 250., 250.] )


    def testReceiverWithData(self):
        data = self.solver.outIntensity(self.mesh1)
        self.solver.inTemperature = data
        self.assertEqual( list(self.solver.inTemperature(self.mesh2)), [200., 200., 400., 400.] )

        self.mesh1.ordering = '01'
        with self.assertRaises(ValueError):
            print(list(self.solver.inTemperature(self.mesh2)))


    def testExternalData(self):
        v = plask.array([[ [1.,10.], [2.,20.] ], [ [3.,30.], [4.,40.] ]])
        self.assertEqual( sys.getrefcount(v), 2 )
        data = plask.Data(v, self.mesh2)
        self.assertEqual( data.dtype, plask.vector2f )
        self.solver.inVectors = data
        self.assertEqual( self.solver.showVectors(), "[1, 5]: [1, 10]\n[3, 5]: [2, 20]\n[1, 15]: [3, 30]\n[3, 15]: [4, 40]\n" )
        self.assertEqual( sys.getrefcount(v), 3 )
        del data
        self.assertEqual( sys.getrefcount(v), 3 )
        self.solver.inVectors = None
        self.assertEqual( sys.getrefcount(v), 2 )


    def testStepProfile(self):
        r1 = geometry.Rectangle(4, 1, None)
        r2 = geometry.Rectangle(4, 2, None)
        stack = geometry.Stack2D()
        h = stack.append(r1)
        stack.append(r2)
        stack.append(r1)
        geom = geometry.Cartesian2D(stack)
        grid = mesh.Rectilinear2D([2.], [0.5, 2.0,  3.5])

        step = StepProfile(geom)
        self.solver.inTemperature = step

        step[r1] = 100.
        self.assertEqual( step[r1], 100. )
        self.assertTrue( self.solver.inTemperature.changed )
        self.assertEqual( list(self.solver.inTemperature(grid)), [100., 300., 100.])
        self.assertFalse( self.solver.inTemperature.changed )

        step[r2] = 400.
        step[r1, h] = 200.
        self.assertTrue( self.solver.inTemperature.changed )
        self.assertEqual( list(self.solver.inTemperature(grid)), [200., 400., 100.])

        del step[r1]
        self.assertEqual( list(self.solver.inTemperature(grid)), [200., 400., 300.])

        self.assertEqual( step.values(), [400., 200.] )
