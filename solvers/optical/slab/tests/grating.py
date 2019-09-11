#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.slab import Fourier2D

plask.config.axes = 'xy'

ns = 3.48
nh = 3.48
nl = 1.00
L = 1.00
tg = 0.20
tl = 0.83
fill = 0.4
N = 12


@material.simple()
class Subs(material.Material):
    nr = ns

@material.simple()
class Hi(material.Material):
    #nr = nh
    def nr(self, lam, temp, conc):
        return nh

@material.simple()
class Lo(material.Material):
    nr = nl


class GratingTest(unittest.TestCase):

    def setUp(self):
        self.stack = geometry.Stack2D()
        bar = geometry.Block2D(fill*L, tg, 'Hi'); self.stack.prepend(bar)
        self.stack.prepend(geometry.Block2D(L, tl, 'Lo'))
        geom = geometry.Cartesian2D(self.stack, bottom='Subs', left='periodic', right='periodic')
        self.solver = Fourier2D('solver')
        self.solver.geometry = geom
        self.solver.set_interface(bar)
        self.solver.size = N
        self.solver.smooth = 0.

    def testComputations(self):
        l_te = array([1500., 1600.])
        self.solver.lam0 = 1500.
        r_te = self.solver.compute_reflectivity(l_te, 'top', 'El')
        self.assertAlmostEqual( r_te[0], 99.934, 2 )
        self.assertAlmostEqual( r_te[1], 98.886, 2 )

        l_tm = array([1298., 1344.])
        self.solver.lam0 = 1300.
        r_tm = self.solver.compute_reflectivity(l_tm, 'top', 'Et')
        self.assertAlmostEqual( r_tm[0], 98.529, 2 )
        self.assertAlmostEqual( r_tm[1], 28.296, 2 )

    def testIntegrals(self):
        self.solver.lam = self.solver.lam0 = 1500.
        scattering = self.solver.scattering('top', 'El')

        box = self.solver.geometry.get_object_bboxes(self.stack)[0]
        msh = mesh.Rectangular2D(mesh.Regular(0., box.right, 501),
                                 mesh.Regular(box.bottom, box.top, 201))
        dx = msh.axis0[1] - msh.axis0[0]
        dy = msh.axis1[1] - msh.axis1[0]
        integral_mesh = msh.elements.mesh

        E = scattering.outLightE(integral_mesh).array
        EE0 = 0.5 * sum(real(E*conj(E))) * dx * dy
        EE1 = scattering.integrateEE(box.bottom, box.top)
        ratio = EE1 / EE0
        print_log('result', "E ratio:", ratio)
        self.assertAlmostEqual(ratio, 1.0, delta=0.1)

        H = scattering.outLightH(integral_mesh).array
        HH0 = 0.5 * sum(real(H*conj(H))) * dx * dy
        HH1 = scattering.integrateHH(box.bottom, box.top)
        ratio = HH1 / HH0
        print_log('result', "H ratio:", ratio)
        self.assertAlmostEqual(ratio, 1.00, delta=0.15)


if __name__ == '__main__':
    unittest.main()
