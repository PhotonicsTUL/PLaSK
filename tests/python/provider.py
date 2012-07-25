#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import plask.mesh
import plasktest




class ReceiverTest(unittest.TestCase):

    def setUp(self):
        self.solver = plasktest.SimpleSolver()
        self.mesh1 = plask.mesh.Regular2D([0., 4., 3], [0., 20., 3])
        self.mesh2 = self.mesh1.getMidpointsMesh();

    def testReceiverWithConstant(self):
        self.solver.inTemperature = 250
        self.assertEqual( list(self.solver.inTemperature(self.mesh2)), [250., 250., 250., 250.] )

    def testReceiverWithData(self):
        data = self.solver.outIntensity(self.mesh1)
        self.solver.inTemperature = data
        self.assertEqual( list(self.solver.inTemperature(self.mesh2)), [200., 200., 400., 400.] )

        self.mesh1.setOrdering("01")
        with self.assertRaises(ValueError):
            print(list(self.solver.inTemperature(self.mesh2)))

    def testExternalData(self):
        v = plask.array([[ [1.,10.], [2.,20.] ], [ [3.,30.], [4.,40.] ]])
        self.assertEqual( v.dtype, plask.vector2f )
        self.solver.inVectors = plask.Data(v, self.mesh2)
        self.assertEqual( self.solver.showVectors(), "[1, 5]: [1, 10]\n[3, 5]: [2, 20]\n[1, 15]: [3, 30]\n[3, 15]: [4, 40]\n" )

