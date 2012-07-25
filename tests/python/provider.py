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