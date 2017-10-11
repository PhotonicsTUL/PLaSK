#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import unittest
import numpy

import plask
import plask.mesh
import plasktest


class ReceiverTest(unittest.TestCase):

    def setUp(self):
        self.solver = plasktest.SimpleSolver()
        self.mesh1 = plask.mesh.Rectangular2D(plask.mesh.Regular(0., 4., 3), plask.mesh.Regular(0., 20., 3))
        self.mesh2 = self.mesh1.get_midpoints();


    def testReceiverWithConstant(self):
        self.solver.inTemperature = 250
        self.assertEqual( list(self.solver.inTemperature(self.mesh2)), [250., 250., 250., 250.] )


    def testReceiverWithData(self):
        data = self.solver.outLightMagnitude(0, self.mesh1)
        self.solver.inTemperature = data
        self.assertEqual( list(self.solver.inTemperature(self.mesh2)), [200., 200., 400., 400.] )

        self.mesh1.ordering = '10'
        with self.assertRaises(Exception):
            print(list(self.solver.inTemperature(self.mesh2)))


    def testExternalData(self):
        v = plask.array([[ [1.,10.], [2.,20.] ], [ [3.,30.], [4.,40.] ]])
        self.assertEqual( sys.getrefcount(v), 2 )
        data = plask.Data(v, self.mesh2)
        self.assertEqual( data.dtype, type(plask.vec(0.,0.)) )
        self.solver.inVectors = data
        self.assertEqual( self.solver.show_vectors(), "[1, 5]: [1, 10]\n[1, 15]: [2, 20]\n[3, 5]: [3, 30]\n[3, 15]: [4, 40]\n" )
                                           #TODO was: "[1, 5]: [1, 10]\n[3, 5]: [2, 20]\n[1, 15]: [3, 30]\n[3, 15]: [4, 40]\n"
        self.assertEqual( sys.getrefcount(v), 3 )
        del data
        self.assertEqual( sys.getrefcount(v), 3 )
        self.solver.inVectors = None
        self.assertEqual( sys.getrefcount(v), 2 )


    def testMultiProviders(self):
       data0 = plask.Data(plask.array([250., 250., 250., 250.]), self.mesh2)
       data1 = plask.Data(plask.array([200., 200., 400., 400.]), self.mesh2)
       self.solver.inIntensity = [data0, data1]
       self.assertEqual( len(self.solver.inIntensity), 2 )
       self.assertEqual( list(self.solver.inIntensity(0,self.mesh2)), list(data0) )
       self.assertEqual( list(self.solver.inIntensity(1,self.mesh2)), list(data1) )

       inout = plasktest.solvers.InOut("inout")
       inout.inWavelength = [1., 2., 3.]
       self.assertEqual( len(inout.inWavelength), 3 )
       self.assertEqual( inout.inWavelength(0), 1. )
       self.assertEqual( inout.inWavelength(1), 2. )
       self.assertEqual( inout.inWavelength(2), 3. )


class PythonProviderTest(unittest.TestCase):

    class CustomSolver(object):
        def __init__(self, parent):
            self.parent = parent
            self.outGain = plask.flow.GainProvider2D(lambda *args: self.get_gain(*args))
            data = plask.Data(numpy.array([1., 2., 3., 4.]), plask.mesh.Rectangular2D([0., 4.], [0., 2.]))
            self.outTemperature = plask.flow.TemperatureProvider2D(data)
        inGain = plask.flow.GainReceiver2D()
        def get_gain(self, deriv, mesh, wavelength, interp):
            self.parent.assertEqual(interp, 'SPLINE')
            ret = wavelength * numpy.array([range(len(mesh)), len(mesh) * [1.]]).T
            return ret

    def setUp(self):
        self.solver = PythonProviderTest.CustomSolver(self)
        self.solver.inGain = self.solver.outGain
        self.binary_solver = plasktest.SimpleSolver()
        self.binary_solver.inTemperature = plask.Data(numpy.array([1., 2., 3., 4.]), plask.mesh.Rectangular2D([0., 4.], [0., 2.]))

    def testSolver(self):
        self.assertEqual( type(self.solver.inGain), plask.flow.GainReceiver2D )
        msh = plask.mesh.Rectangular2D(plask.mesh.Regular(0.,1., 2), plask.mesh.Regular(0.,1., 3))
        res = self.solver.inGain(msh, 10., 'spline')
        print(res.dtype, list(res))
        self.assertEqual( [r[0] for r in res], [0., 10., 20., 30., 40., 50.] )
        self.assertEqual( [r[1] for r in res], 6 * [10.] )
        msh = plask.mesh.Rectangular2D([2.], [1.])
        self.assertEqual( list(self.solver.outTemperature(msh, 'linear')), [2.5] )
        self.assertEqual( list(self.binary_solver.inTemperature(msh, 'linear')), [2.5] )

    class CustomData(object):
        def __getitem__(self, i):
            return 10. * (i + 1)

    def testCustomData(self):
        provider = plask.flow.TemperatureProvider2D(lambda *args: PythonProviderTest.CustomData())
        msh = plask.mesh.Rectangular2D([0., 1., 2.], [0.])
        self.assertEqual( list(provider(msh)), [10., 20., 30.] )


class DataTest(unittest.TestCase):

    def testOperations(self):
        v = plask.array([[ [1.,10.], [2.,20.] ], [ [3.,30.], [4.,40.] ]])
        data = plask.Data(v, plask.mesh.Rectangular2D(plask.mesh.Regular(0., 4., 3), plask.mesh.Regular(0., 20., 3)).get_midpoints())
        self.assertEqual( data + data, 2 * data )


class ProfileTest(unittest.TestCase):

    def testProfile(self):
        hot = plask.geometry.Rectangle(20, 2, 'GaAs')
        cold = plask.geometry.Rectangle(20, 10, 'GaAs')
        stack = plask.geometry.Stack2D()
        stack.prepend(hot)
        stack.prepend(cold)
        geom = plask.geometry.Cylindrical2D(stack)
        profile = plask.StepProfile(geom)
        profile[hot] = 1e7
        receiver = plask.flow.HeatReceiverCyl()
        receiver.connect(profile.outHeat)
        self.assertEqual( list(receiver(mesh.Rectangular2D(mesh.Ordered([10]), mesh.Ordered([5, 11])))), [0., 1e7] )
        profile[hot] = 2e7
        self.assertEqual( list(receiver(mesh.Rectangular2D(mesh.Ordered([10]),  mesh.Ordered([5, 11])))), [0., 2e7] )

    def testAdding(self):
        warm = plask.geometry.Rectangle(20, 2, 'GaAs')
        hot = plask.geometry.Rectangle(20, 2, 'GaAs')
        stack = plask.geometry.Stack2D()

        top = stack.prepend(warm)
        stack.prepend(hot)
        bottom = stack.prepend(warm)

        geom = plask.geometry.Cylindrical2D(stack)

        profile1 = plask.StepProfile(geom)
        profile1[warm, top] = 1e9
        profile1[warm, top] = 1e7
        profile1[hot] = 1e9
        profile1[hot] = 1e7

        profile2 = plask.StepProfile(geom)
        profile2[warm, bottom] = 1e7
        profile2[hot] = 1e7

        receiver = plask.flow.HeatReceiverCyl()
        receiver.connect(profile1.outHeat + profile2.outHeat)

        self.assertEqual( list(receiver(mesh.Rectangular2D(mesh.Ordered([10]), mesh.Ordered([1, 3, 5])))), [1e7, 2e7, 1e7] )


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
