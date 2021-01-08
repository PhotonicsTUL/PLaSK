#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *
from numpy.testing import assert_array_equal, assert_array_almost_equal

from plask import *
from plask import material, geometry, mesh
from plask.geometry import Cartesian2D, Cartesian3D, Cylindrical

from optical.slab import *
from electrical.diffusion import Diffusion2D, DiffusionCyl


@material.simple('GaAs')
class Conc(material.Material):

    def A(self, T=300):
        return 30000000

    def B(self, T=300):
        return 1.7e-10

    def C(self, T=300):
        return 6e-30

    def D(self, T=300):
        return 10

    def nr(self, lam, T, n):
        return 3.0 + 1e-19 * n


class CarriersTest:
    name = None
    Geometry = None
    Solver = None
    size = 0
    Diffusion = None
    geometry_kwargs = {}

    def setUp(self):
        clad = geometry.Rectangle(0.500, 0.100, Conc())
        qw = geometry.Rectangle(0.500, 0.002, Conc())
        qw.role = 'QW'
        ba = geometry.Rectangle(0.500, 0.001, Conc())
        active = geometry.Stack2D()
        active.role = 'active'
        active.prepend(qw)
        active.prepend(ba)
        active.prepend(qw)
        active.prepend(ba)
        active.prepend(qw)
        stack = geometry.Stack2D()
        stack.prepend(clad)
        stack.prepend(active)
        stack.prepend(clad)
        self.geometry = self.Geometry(stack, **self.geometry_kwargs)
        self.solver = self.Solver(self.name)
        self.solver.geometry = self.geometry
        self.solver.lam0 = 1000.
        self.solver.size = self.size
        self.solver.refine = 0
        self.solver.inGain = 1000.
        self.test_mesh = mesh.Rectangular2D([0.], [0.050, 0.104, 0.158])

    def test_no_carriers(self):
        nr = self.solver.outRefractiveIndex(self.test_mesh).array[...,0].flatten().real
        assert_array_equal(nr, [3.0, 3.0, 3.0])

    def test_carriers(self):
        diffusion = self.Diffusion('diffusion')
        diffusion.geometry = self.geometry
        diffusion.inCurrentDensity = vec(0., 10.)
        diffusion.compute_threshold()
        cc = diffusion.outCarriersConcentration(mesh.Rectangular2D([0.], [0.104]))[0]
        self.assertLess(1e18, cc)

        self.solver.inCarriersConcentration = diffusion.outCarriersConcentration
        nr = self.solver.outRefractiveIndex(self.test_mesh).array[...,0].flatten().real
        assert_array_almost_equal(nr, [3.0, 3.0 + 1e-19 * cc, 3.0])


class Fourier2DCarriers(CarriersTest, unittest.TestCase):
    name = "fourier2"
    Geometry = Cartesian2D
    Solver = Fourier2D
    Diffusion = Diffusion2D
    geometry_kwargs = {'left': 'mirror', 'right': 'periodic'}


class BesselCylCarriers(CarriersTest, unittest.TestCase):
    name = "bessel"
    Geometry = Cylindrical
    Solver = BesselCyl
    size = 1
    Diffusion = DiffusionCyl


class Fourier3DCarriers(CarriersTest, unittest.TestCase):
    name = "fourier3"
    Solver = Fourier3D
    Diffusion = DiffusionCyl
    geometry_kwargs = {'left': 'mirror', 'back': 'mirror', 'right': 'periodic', 'front': 'periodic'}

    def Geometry(self, item, **kwargs):
        return Cartesian3D(geometry.Clip3D(geometry.Revolution(item), left=0., back=0., right=0.35, front=0.35), **kwargs)

    def setUp(self):
        CarriersTest.setUp(self)
        self.geometry = Cylindrical(self.geometry.item.item)
        self.test_mesh = mesh.Rectangular3D([0.], [0.], [0.050, 0.104, 0.158])
