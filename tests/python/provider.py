#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import unittest
import numpy

import plask
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
        data = self.solver.outLightIntensity(0,self.mesh1)
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
            self._receiver = plask.flow.GainReceiver2D()
        inGain = property(lambda self: self._receiver, lambda self,provider: self._receiver.connect(provider))
        def get_gain(self, mesh, wavelength, interp):
            self.parent.assertEqual(interp, plask.interpolation.SPLINE)
            return wavelength * numpy.arange(len(mesh))

    def setUp(self):
        self.solver = PythonProviderTest.CustomSolver(self)
        self.solver.inGain = self.solver.outGain

    def testAll(self):
        self.assertEqual( type(self.solver.inGain), plask.flow.GainReceiver2D )
        msh = plask.mesh.Regular2D((0.,1., 2), (0.,1., 3))
        res = self.solver.inGain(msh, 10., 'spline')
        self.assertEqual(list(res), [0., 10., 20., 30., 40., 50.])


class DataTest(unittest.TestCase):

    def testOperations(self):
        v = plask.array([[ [1.,10.], [2.,20.] ], [ [3.,30.], [4.,40.] ]]).transpose((1,0,2))
        data = plask.Data(v, plask.mesh.Regular2D((0., 4., 3), (0., 20., 3)).get_midpoints())
        self.assertEqual( data + data, 2 * data )
