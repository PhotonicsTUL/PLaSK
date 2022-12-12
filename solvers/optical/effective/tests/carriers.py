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
from numpy.testing import assert_array_equal, assert_array_almost_equal

from plask import *
from plask import material, geometry, mesh
from plask.geometry import Cartesian2D, Cylindrical

from optical.effective import EffectiveIndex2D, EffectiveFrequencyCyl
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
    Diffusion = None
    geometry_kwargs = {}

    def setUp(self):
        clad = geometry.Rectangle(0.500, 0.100, Conc())
        qw = geometry.Rectangle(0.500, 0.006, Conc())
        qw.roles = 'QW', 'active'
        stack = geometry.Stack2D()
        stack.prepend(clad)
        stack.prepend(qw)
        stack.prepend(clad)
        self.geometry = self.Geometry(stack, **self.geometry_kwargs)
        self.solver = self.Solver(self.name)
        self.solver.geometry = self.geometry
        self.solver.inGain = 1000.
        self.test_mesh = mesh.Rectangular2D([0.], [0.050, 0.103, 0.156])

    def test_no_carriers(self):
        nr = self.solver.outRefractiveIndex(self.test_mesh).array[0,:,0].real
        assert_array_equal(nr, [3.0, 3.0, 3.0])

    def test_carriers(self):
        diffusion = self.Diffusion('diffusion')
        diffusion.geometry = self.geometry
        diffusion.inCurrentDensity = vec(0., 10.)
        diffusion.compute_threshold()
        cc = diffusion.outCarriersConcentration(mesh.Rectangular2D([0.250], [0.103]))[0]
        self.assertLess(1e18, cc)

        self.solver.inCarriersConcentration = diffusion.outCarriersConcentration
        nr = self.solver.outRefractiveIndex(self.test_mesh).array[0,:,0].real
        assert_array_almost_equal(nr, [3.0, 3.0 + 1e-19 * cc, 3.0])


class EffectiveIndexCarriers(CarriersTest, unittest.TestCase):
    name = "eim"
    Geometry = Cartesian2D
    Solver = EffectiveIndex2D
    Diffusion = Diffusion2D
    geometry_kwargs = {'left': 'mirror'}


class EffectiveFrequencyCarriers(CarriersTest, unittest.TestCase):
    name = "efm"
    Geometry = Cylindrical
    Solver = EffectiveFrequencyCyl
    Diffusion = DiffusionCyl
