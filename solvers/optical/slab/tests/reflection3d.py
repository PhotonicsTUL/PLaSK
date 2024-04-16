#!/usr/bin/env python3
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

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.slab import Fourier3D, PML


@material.simple()
class Mat(material.Material):
    nr = 2.


class Reflection3D_Test(unittest.TestCase):

    def setUp(self):
        config.axes = 'xyz'
        background = geometry.Cuboid(1.0, 1.0, 0.2, None)
        obj = geometry.Cuboid(0.5, 0.5, 0.2, Mat())
        align = geometry.AlignContainer3D()
        align.append(background, back=0., left=0., bottom=0.)
        align.append(obj, back=0., left=0., bottom=0.)
        geom = geometry.Cartesian3D(align, back='periodic', front='periodic', left='periodic', right='periodic')
        self.solver = Fourier3D()
        self.solver.geometry = geom
        self.solver.set_interface(0.5)
        self.solver.wavelength = 1000.
        self.solver.smooth = 0.
        self.solver.size = 11, 11        # number of material coefficients in each direction 4*11+1 = 45
        self.solver.refine = 16, 16
        self.solver.rule = 'direct'

    # 2.5 = ( 1**2 + 2**2 ) / 2
    # 1.6 = 2 / ( 1**(-2) + 2**(-2) )

    def testTran(self):
        msh_tran = mesh.Rectangular3D(mesh.Ordered([0.25]), mesh.Regular(0, 1, 46), mesh.Ordered([0.1]))
        prof_tran = self.solver.outEpsilon(msh_tran, 'nearest').array
        self.assertAlmostEqual(prof_tran[0,0,0,0,0], 2.5, 5)
        self.assertAlmostEqual(prof_tran[0,0,0,1,1], 1.6, 5)
        self.assertAlmostEqual(prof_tran[0,0,0,2,2], 2.5, 5)
        self.assertAlmostEqual(prof_tran[0,1,0,0,0], 4., 5)
        self.assertAlmostEqual(prof_tran[0,1,0,1,1], 4., 5)
        self.assertAlmostEqual(prof_tran[0,1,0,2,2], 4., 5)
        self.assertAlmostEqual(prof_tran[0,-2,0,0,0], 1., 5)
        self.assertAlmostEqual(prof_tran[0,-2,0,1,1], 1., 5)
        self.assertAlmostEqual(prof_tran[0,-2,0,2,2], 1., 5)
        self.assertAlmostEqual(prof_tran[0,22,0,0,0], 4., 5)
        self.assertAlmostEqual(prof_tran[0,22,0,1,1], 4., 5)
        self.assertAlmostEqual(prof_tran[0,22,0,2,2], 4., 5)
        self.assertAlmostEqual(prof_tran[0,23,0,0,0], 1., 5)
        self.assertAlmostEqual(prof_tran[0,23,0,1,1], 1., 5)
        self.assertAlmostEqual(prof_tran[0,23,0,2,2], 1., 5)
        print("{} {} {}".format(*prof_tran[0, [0,1,-2], 0, :3].real))
        # figure()
        # plot(msh_tran.axis1, prof_tran[0,:,0,0].real, '.')
        # plot(msh_tran.axis1, prof_tran[0,:,0,1].real, '.')
        # plot(msh_tran.axis1, prof_tran[0,:,0,2].real, '.')
        # xlabel(u'$x$ (µm)')
        # ylabel(u'$n_r$')

    def testLong(self):
        msh_long = mesh.Rectangular3D(mesh.Regular(0, 1, 46), mesh.Ordered([0.25]), mesh.Ordered([0.1]))
        prof_long = self.solver.outEpsilon(msh_long, 'nearest').array
        self.assertAlmostEqual(prof_long[0,0,0,0,0], 1.6, 5)
        self.assertAlmostEqual(prof_long[0,0,0,1,1], 2.5, 5)
        self.assertAlmostEqual(prof_long[0,0,0,2,2], 2.5, 5)
        self.assertAlmostEqual(prof_long[1,0,0,0,0], 4., 5)
        self.assertAlmostEqual(prof_long[1,0,0,1,1], 4., 5)
        self.assertAlmostEqual(prof_long[1,0,0,2,2], 4., 5)
        self.assertAlmostEqual(prof_long[-2,0,0,0,0], 1., 5)
        self.assertAlmostEqual(prof_long[-2,0,0,1,1], 1., 5)
        self.assertAlmostEqual(prof_long[-2,0,0,2,2], 1., 5)
        self.assertAlmostEqual(prof_long[22,0,0,0,0], 4., 5)
        self.assertAlmostEqual(prof_long[22,0,0,1,1], 4., 5)
        self.assertAlmostEqual(prof_long[22,0,0,2,2], 4., 5)
        self.assertAlmostEqual(prof_long[23,0,0,0,0], 1., 5)
        self.assertAlmostEqual(prof_long[23,0,0,1,1], 1., 5)
        self.assertAlmostEqual(prof_long[23,0,0,2,2], 1., 5)
        print("{} {} {}".format(*prof_long[[0,1,-2], 0, 0, :3].real))
        # figure()
        # plot(msh_long.axis0, prof_long[:,0,0,0].real, '.')
        # plot(msh_long.axis0, prof_long[:,0,0,1].real, '.')
        # plot(msh_long.axis0, prof_long[:,0,0,2].real, '.')
        # xlabel(u'$z$ (µm)')
        # ylabel(u'$n_r$')


    def testWrappers(self):
        self.solver.size = 2, 3
        self.solver.refine = 5, 6

        self.assertEqual(str(self.solver.size), "(2, 3)")

        self.assertEqual(self.solver.size.x, 2)
        self.assertEqual(self.solver.size[1], 3)

        self.solver.size[0] = 4
        self.assertEqual(self.solver.size.x, 4)

        self.solver.wavelength = 1000.
        self.solver.outEpsilon(mesh.Rectangular3D([0.], [0.25], [0.1]))
        self.assertEqual(self.solver.initialized, True)
        self.solver.size = 5
        self.assertEqual(self.solver.initialized, False)
        self.assertEqual(str(self.solver.size), "(5, 5)")

        self.solver.outEpsilon(mesh.Rectangular3D([0.], [0.25], [0.1]))
        self.assertEqual(self.solver.initialized, True)
        self.solver.refine[1] = 7
        self.assertEqual(self.solver.refine.y, 7)
        self.assertEqual(self.solver.initialized, False)

        print (self.solver.pmls)
        self.solver.pmls = PML(1-2j, 1.0, 0.5)
        self.assertEqual(self.solver.pmls[0].size, 1.)

        self.assertEqual(self.solver.initialized, False)
        self.solver.outEpsilon(mesh.Rectangular3D([0.], [0.25], [0.1]))
        self.assertEqual(self.solver.initialized, True)
        self.solver.pmls[1].size = 2.
        self.assertEqual(self.solver.pmls.y.size, 2.)
        self.assertEqual(self.solver.initialized, False)


if __name__ == '__main__':
    unittest.main()
