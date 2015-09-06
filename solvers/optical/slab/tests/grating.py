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
    @staticmethod
    def nr(): return ns

@material.simple()
class Hi(material.Material):
    @staticmethod
    def nr(): return nh

@material.simple()
class Lo(material.Material):
    @staticmethod
    def nr(): return nl


class GratingTest(unittest.TestCase):

    def setUp(self):
        stack = geometry.Stack2D()
        bar = geometry.Block2D(fill*L, tg, 'Hi'); stack.prepend(bar)
        stack.prepend(geometry.Block2D(L, tl, 'Lo'))
        geom = geometry.Cartesian2D(stack, bottom='Subs', left='periodic', right='periodic')
        self.solver = Fourier2D('solver')
        self.solver.geometry = geom
        self.solver.set_interface(bar)
        self.solver.size = N
        self.solver.smooth = 0.

    def testComputations(self):
        l_te = array([1500., 1600.])
        self.solver.lam0 = 1500.
        r_te = self.solver.compute_reflectivity(l_te, 'El', 'top')
        self.assertAlmostEqual( r_te[0], 99.934, 2 )
        self.assertAlmostEqual( r_te[1], 98.886, 2 )

        l_tm = array([1298., 1344.])
        self.solver.lam0 = 1500.
        r_tm = self.solver.compute_reflectivity(l_tm, 'Et', 'top')
        self.assertAlmostEqual( r_tm[0], 99.549, 2 )
        self.assertAlmostEqual( r_tm[1], 26.479, 2 )
