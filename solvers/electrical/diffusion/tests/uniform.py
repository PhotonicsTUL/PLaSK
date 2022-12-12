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
from numpy.testing import assert_allclose

from plask import *
from plask import material, geometry, mesh
from plask.geometry import Cartesian2D, Cylindrical

from electrical.diffusion import Diffusion2D, DiffusionCyl


@material.simple('GaAs')
class GaAsQW(material.Material):

    def A(self, T=300):
        return 3e7

    def B(self, T=300):
        return 1.7e-10

    def C(self, T=300):
        return 6e-30

    def D(self, T=300):
        return 10


class DiffusionTest:
    name = None
    Geometry = None
    Solver = None
    geometry_kwargs = {}

    def setUp(self):
        clad = geometry.Rectangle(0.500, 0.100, 'GaAs')
        qw = geometry.Rectangle(0.500, 0.002, 'GaAsQW')
        qw.role = 'QW'
        ba = geometry.Rectangle(0.500, 0.001, 'GaAs')
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
        self.solver.accuracy = 0.0001
        # self.solver.fem_method = 'linear'
        self.test_mesh = mesh.Rectangular2D(linspace(-0.4, 0.4, 9), [0.104])

        self.n = 1.0e19
        self.j = 1e-7 * (qw.material.A() * self.n + qw.material.B() * self.n**2 + qw.material.C() * self.n**3) * (phys.qe * 0.006)

    def test_diffusion(self):
        self.solver.inCurrentDensity = vec(0., self.j)
        self.solver.compute_threshold()
        n = array(self.solver.outCarriersConcentration(self.test_mesh))
        assert_allclose(n, 9 * [self.n])


class Diffusion2D(DiffusionTest, unittest.TestCase):
    name = "diffusion2d"
    Geometry = Cartesian2D
    Solver = Diffusion2D
    geometry_kwargs = {'left': 'mirror', 'right': 'air'}


class DiffusionCyl(DiffusionTest, unittest.TestCase):
    name = "diffusioncyl"
    Geometry = Cylindrical
    Solver = DiffusionCyl
