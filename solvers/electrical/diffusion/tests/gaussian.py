# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2023 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import unittest

from numpy import *
from numpy.testing import assert_allclose

from plask import *
from plask import material, geometry, mesh
from plask.geometry import Cartesian2D, Cylindrical
from plask.flow import CurrentDensityProvider2D, CurrentDensityProviderCyl

from electrical.diffusion import Diffusion2D, DiffusionCyl

A = 3e7  # [1/s]
B = 1.7e-10  # [cm³/s]
C = 6e-27  # [cm⁶/s]
D = 10.  # [cm²/s]

L = 4.0


@material.simple('GaAs')
class GaAsQW(material.Material):

    def A(self, T=300):
        return A

    def B(self, T=300):
        return B

    def C(self, T=300):
        return C

    def D(self, T=300):
        return D


class DiffusionTest:
    name = None
    Geometry = None
    Solver = None
    geometry_kwargs = {}
    provider = None
    axes = None

    def setUp(self):
        config.axes = self.axes
        clad = geometry.Rectangle(L, 0.100, 'GaAs')
        qw = geometry.Rectangle(L, 0.002, 'GaAsQW')
        qw.role = 'QW'
        ba = geometry.Rectangle(L, 0.001, 'GaAs')
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
        self.solver.maxerr = 0.0001
        self.solver.inCurrentDensity = self.provider(lambda m, _: array([zeros(len(m.axis0)), self.j(array(m.axis0))]).T)
        self.test_mesh = mesh.Rectangular2D(linspace(-3.5, 3.5, 15), [0.104])

    def n(self, x):
        return 1e19 * (exp(-x**2) + 0.5)

    def d2n(self, x):
        return 2e19 * (2 * x**2 - 1) * exp(-x**2)

    def j(self, x):
        n = self.n(x)
        nj = 1e8 * D * self.d2n(x) - A * n - B * n**2 - C * n**3
        return -1e-7 * (phys.qe * 0.006) * nj

    def test_diffusion(self):
        self.solver.compute()
        n = array(self.solver.outCarriersConcentration(self.test_mesh))
        assert_allclose(n, self.n(array(self.test_mesh.axis0)), 1e-5)


class Diffusion2D(DiffusionTest, unittest.TestCase):
    name = "diffusion2d"
    Geometry = Cartesian2D
    geometry_kwargs = {'left': 'mirror', 'right': 'air'}
    Solver = Diffusion2D
    provider = CurrentDensityProvider2D
    axes = 'xy'


class DiffusionCyl(DiffusionTest, unittest.TestCase):
    name = "diffusioncyl"
    Geometry = Cylindrical
    Solver = DiffusionCyl
    provider = CurrentDensityProviderCyl
    axes = 'rz'

    def d2n(self, x):
        return 4e19 * (x**2 - 1) * exp(-x**2)


if __name__ == '__main__':
    for Test in Diffusion2D, DiffusionCyl:
        try:
            figure()
            test = Test()
            test.setUp()
            x = linspace(-4.1, 4.1, 821)
            axhline(0., lw=0.7, color='k')
            plot(x, test.j(x), color='C0', label='current')
            xlabel(f"${config.axes[1]}$ [µm]")
            ylabel("$j$ [kA/cm]")
            legend(loc='upper left')
            twinx()
            plot(x, test.n(x), 'C1', label='concentration (analytic)')
            test.solver.compute()
            plot_profile(
                test.solver.outCarriersConcentration(mesh.Rectangular2D(x, [0.104])),
                color='C2',
                ls='--',
                label='concentration (numeric)'
            )
            xlabel(f"${config.axes[1]}$ [µm]")
            ylabel("$n$ [cm$^{-3}$]")
            legend(loc='upper right')
            window_title(test.Solver.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
    show()
