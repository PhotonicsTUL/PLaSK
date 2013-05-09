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

        self.mesh1.ordering = '10'
        with self.assertRaises(ValueError):
            print(list(self.solver.inTemperature(self.mesh2)))


    def testExternalData(self):
        v = plask.array([[ [1.,10.], [2.,20.] ], [ [3.,30.], [4.,40.] ]]).transpose((1,0,2))
        self.assertEqual( sys.getrefcount(v), 2 )
        data = plask.Data(v, self.mesh2)
        self.assertEqual( data.dtype, plask.vector2f )
        self.solver.inVectors = data
        self.assertEqual( self.solver.show_vectors(), "[1, 5]: [1, 10]\n[3, 5]: [2, 20]\n[1, 15]: [3, 30]\n[3, 15]: [4, 40]\n" )
        self.assertEqual( sys.getrefcount(v), 3 )
        del data
        self.assertEqual( sys.getrefcount(v), 3 )
        self.solver.inVectors = None
        self.assertEqual( sys.getrefcount(v), 2 )


    def testStepProfile(self):
        r1 = plask.geometry.Rectangle(4, 1, None)
        r2 = plask.geometry.Rectangle(4, 2, None)
        stack = plask.geometry.Stack2D()
        h = stack.append(r1)
        stack.append(r2)
        stack.append(r1)
        geom = plask.geometry.Cartesian2D(stack)
        grid = plask.mesh.Rectilinear2D([2.], [0.5, 2.0,  3.5])

        step = plask.StepProfile(geom)
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



class PythonProviderTest(unittest.TestCase):

    class CustomSolver(object):
        def __init__(self, parent):
            self.parent = parent
            self.outGain = ProviderForGain2D(lambda *args: self.get_gain(*args))
            self.receiver = ReceiverForGain2D()
        inGain = property(lambda self: self.receiver, lambda self,provider: self.receiver.connect(provider))
        def get_gain(self, mesh, wavelength, interpolation):
            self.parent.assertEqual(interpolation, interpolation.SPLINE)
            return Data(wavelength * arange(len(mesh)), mesh)

    def setUp(self):
        self.solver = PythonProviderTest.CustomSolver(self)
        self.solver.inGain = self.solver.outGain

    def testAll(self):
        msh = mesh.Regular2D((0.,1., 2), (0.,1., 3))
        res = self.solver.inGain(msh, 10., 'spline')
        self.assertEqual(list(res), [0., 10., 20., 30., 40., 50.])
