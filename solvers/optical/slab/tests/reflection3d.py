#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.slab import FourierReflection3D, PML


@material.simple
class Mat(material.Material):
    @staticmethod
    def nr():
        return 2.


class Averaging_Test(unittest.TestCase):

    def setUp(self):
        background = geometry.Cuboid(1.0, 1.0, 0.2, None)
        obj = geometry.Cuboid(0.5, 0.5, 0.2, Mat())
        align = geometry.AlignContainerVert3D(bottom=0.)
        align.add(background, back=0., left=0.)
        align.add(obj, back=0., left=0.)
        geom = geometry.Cartesian3D(align, back='periodic', front='periodic', left='periodic', right='periodic')
        self.solver = FourierReflection3D()
        self.solver.geometry = geom
        self.solver.wavelength = 1000.
        self.solver.smooth = 0.
        self.solver.size = 11, 11        # number of material coefficients in each direction 4*11+1 = 45
        self.solver.refine = 16, 16

    # 2.5 = ( 1**2 + 2**2 ) / 2
    # 1.6 = 2 / ( 1**(-2) + 2**(-2) )

    def testTran(self):
        msh_tran = mesh.Rectangular3D(mesh.Ordered([0.25]), mesh.Regular(0, 1, 46), mesh.Ordered([0.1]))
        prof_tran = self.solver.get_refractive_index_profile(msh_tran, 'nearest')
        self.assertAlmostEqual( prof_tran.array[0,0,0,0], sqrt(2.5), 5 )
        self.assertAlmostEqual( prof_tran.array[0,0,0,1], sqrt(1.6), 5 )
        self.assertAlmostEqual( prof_tran.array[0,0,0,2], sqrt(2.5), 5 )
        self.assertAlmostEqual( prof_tran.array[0,1,0,0], 2., 5 )
        self.assertAlmostEqual( prof_tran.array[0,1,0,1], 2., 5 )
        self.assertAlmostEqual( prof_tran.array[0,1,0,2], 2., 5 )
        self.assertAlmostEqual( prof_tran.array[0,-2,0,0], 1., 5 )
        self.assertAlmostEqual( prof_tran.array[0,-2,0,1], 1., 5 )
        self.assertAlmostEqual( prof_tran.array[0,-2,0,2], 1., 5 )
        self.assertAlmostEqual( prof_tran.array[0,22,0,0], 2., 5 )
        self.assertAlmostEqual( prof_tran.array[0,22,0,1], 2., 5 )
        self.assertAlmostEqual( prof_tran.array[0,22,0,2], 2., 5 )
        self.assertAlmostEqual( prof_tran.array[0,23,0,0], 1., 5 )
        self.assertAlmostEqual( prof_tran.array[0,23,0,1], 1., 5 )
        self.assertAlmostEqual( prof_tran.array[0,23,0,2], 1., 5 )
        # print("{} {} {}".format(*prof_tran.array[0, [0,1,-2], 0, :3].real))
        # figure()
        # plot(msh_tran.axis1, prof_tran.array[0,:,0,0].real, '.')
        # plot(msh_tran.axis1, prof_tran.array[0,:,0,1].real, '.')
        # plot(msh_tran.axis1, prof_tran.array[0,:,0,2].real, '.')
        # xlabel(u'$x$ [µm]')
        # ylabel(u'$n_r$')

    def testLong(self):
        msh_long = mesh.Rectangular3D(mesh.Regular(0, 1, 46), mesh.Ordered([0.25]), mesh.Ordered([0.1]))
        prof_long = self.solver.get_refractive_index_profile(msh_long, 'nearest')
        self.assertAlmostEqual( prof_long.array[0,0,0,0], sqrt(1.6), 5 )
        self.assertAlmostEqual( prof_long.array[0,0,0,1], sqrt(2.5), 5 )
        self.assertAlmostEqual( prof_long.array[0,0,0,2], sqrt(2.5), 5 )
        self.assertAlmostEqual( prof_long.array[1,0,0,0], 2., 5 )
        self.assertAlmostEqual( prof_long.array[1,0,0,1], 2., 5 )
        self.assertAlmostEqual( prof_long.array[1,0,0,2], 2., 5 )
        self.assertAlmostEqual( prof_long.array[-2,0,0,0], 1., 5 )
        self.assertAlmostEqual( prof_long.array[-2,0,0,1], 1., 5 )
        self.assertAlmostEqual( prof_long.array[-2,0,0,2], 1., 5 )
        self.assertAlmostEqual( prof_long.array[22,0,0,0], 2., 5 )
        self.assertAlmostEqual( prof_long.array[22,0,0,1], 2., 5 )
        self.assertAlmostEqual( prof_long.array[22,0,0,2], 2., 5 )
        self.assertAlmostEqual( prof_long.array[23,0,0,0], 1., 5 )
        self.assertAlmostEqual( prof_long.array[23,0,0,1], 1., 5 )
        self.assertAlmostEqual( prof_long.array[23,0,0,2], 1., 5 )
        # print("{} {} {}".format(*prof_long.array[[0,1,-2], 0, 0, :3].real))
        # figure()
        # plot(msh_long.axis0, prof_long.array[:,0,0,0].real, '.')
        # plot(msh_long.axis0, prof_long.array[:,0,0,1].real, '.')
        # plot(msh_long.axis0, prof_long.array[:,0,0,2].real, '.')
        # xlabel(u'$z$ [µm]')
        # ylabel(u'$n_r$')


class Solver_Test(unittest.TestCase):

    def setUp(self):
        config.axes = 'xyz'
        background = geometry.Cuboid(1.0, 1.0, 0.2, None)
        geom = geometry.Cartesian3D(background, back='periodic', front='periodic', left='periodic', right='periodic')
        self.solver = FourierReflection3D()
        self.solver.geometry = geom


    def testWrappers(self):
        self.solver.size = 2, 3
        self.solver.refine = 5, 6

        self.assertEqual( str(self.solver.size), "(2, 3)" )

        self.assertEqual( self.solver.size.x, 2 )
        self.assertEqual( self.solver.size[1], 3 )

        self.solver.size[0] = 4
        self.assertEqual( self.solver.size.x, 4 )

        self.solver.wavelength = 1000.
        self.solver.get_refractive_index_profile(mesh.Rectangular3D([0.], [0.25], [0.1]))
        self.assertEqual( self.solver.initialized, True )
        self.solver.size = 5
        self.assertEqual( self.solver.initialized, False )
        self.assertEqual( str(self.solver.size), "(5, 5)" )

        self.solver.get_refractive_index_profile(mesh.Rectangular3D([0.], [0.25], [0.1]))
        self.assertEqual( self.solver.initialized, True )
        self.solver.refine[1] = 7
        self.assertEqual( self.solver.refine.y, 7 )
        self.assertEqual( self.solver.initialized, False )

        print (self.solver.pmls)
        self.solver.pmls = PML(1-2j, 1.0, 0.5)
        self.assertEqual( self.solver.pmls[0].size, 1. )

        self.assertEqual( self.solver.initialized, False )
        self.solver.get_refractive_index_profile(mesh.Rectangular3D([0.], [0.25], [0.1]))
        self.assertEqual( self.solver.initialized, True )
        self.solver.pmls[1].size = 2.
        self.assertEqual( self.solver.pmls.y.size, 2. )
        self.assertEqual( self.solver.initialized, False )
