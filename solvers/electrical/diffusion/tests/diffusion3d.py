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

from plask.flow import CurrentDensityProvider3D

from electrical.diffusion import Diffusion3D

config.loglevel = 'debug'
config.axes = 'xyz'
np.set_printoptions(linewidth=132)

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


class DiffusionTest(unittest.TestCase):

    def setUp(self):
        # clad = geometry.Cuboid(L, L, 0.100, 'GaAs')
        # qw = geometry.Cuboid(L, L, 0.002, 'GaAsQW')
        # ba = geometry.Cuboid(L, L, 0.001, 'GaAs')
        clad = geometry.Cylinder(L, 0.100, 'GaAs')
        qw = geometry.Cylinder(L, 0.002, 'GaAsQW')
        ba = geometry.Cylinder(L, 0.001, 'GaAs')
        qw.role = 'QW'
        active = geometry.Stack3D(x=0, y=0)
        active.role = 'active'
        active.prepend(qw)
        active.prepend(ba)
        active.prepend(qw)
        active.prepend(ba)
        active.prepend(qw)
        stack = geometry.Stack3D(x=0, y=0)
        stack.prepend(clad)
        stack.prepend(active)
        stack.prepend(clad)
        clip = geometry.Clip3D(stack, left=0., back=0.)
        self.geometry = geometry.Cartesian3D(clip, left='mirror', right='air', back='mirror', front='air')
        self.solver = Diffusion3D("diffusion3d")
        # self.solver.algorithm = 'iterative'
        self.solver.geometry = self.geometry
        self.solver.mesh = mesh.Rectangular3D.RegularGenerator(0.01 * L, 0.01 * L, 0.01)
        self.solver.maxerr = 0.0001
        self.qwm = qw.material
        self.test_mesh = mesh.Rectangular3D(linspace(-0.8 * L, 0.8 * L, 5), linspace(-0.8 * L, 0.8 * L, 5), [0.104])

    def filter(self, values, x, y):
        mask = x**2 + y**2 <= L**2
        values[~mask] = 0.
        return values

    def test_uniform(self):
        n = 1.0e19
        j = 1e-7 * (self.qwm.A() * n + self.qwm.B() * n**2 + self.qwm.C() * n**3) * (phys.qe * 0.006)
        self.solver.inCurrentDensity = vec(0., 0., j)
        self.solver.compute()
        res = self.solver.outCarriersConcentration(self.test_mesh).array[:, :, 0]
        ref = n * ones_like(res)
        y, x = meshgrid(self.test_mesh.axis0, self.test_mesh.axis1)
        assert_allclose(res, self.filter(ref, x, y), 1e-5)

    def n(self, p):
        x, y, z = p.T
        return self.filter(1e19 * (exp(-x**2) * exp(-y**2) + 0.5), x, y)

    def d2nx(self, p):
        x, y, z = p.T
        return self.filter(2e19 * (2 * x**2 - 1) * exp(-x**2) * exp(-y**2), x, y)

    def d2ny(self, p):
        x, y, z = p.T
        return self.filter(2e19 * exp(-x**2) * (2 * y**2 - 1) * exp(-y**2), x, y)

    def j(self, p):
        x, y, z = p.T
        n = self.n(p)
        nj = 1e8 * D * (self.d2nx(p) + self.d2ny(p)) - A * n - B * n**2 - C * n**3
        return self.filter(-1e-7 * (phys.qe * 0.006) * nj, x, y)

    def test_gaussian(self):
        self.solver.inCurrentDensity = CurrentDensityProvider3D(
            lambda m, _: array([zeros(len(m)), zeros(len(m)), self.j(array(m))]).T
        )
        self.solver.compute()
        n = array(self.solver.outCarriersConcentration(self.test_mesh))
        assert_allclose(n, self.n(array(self.test_mesh)), 0.5e-3)


if __name__ == '__main__':
    test = DiffusionTest()
    test.setUp()

    x = linspace(-1.1 * L, 1.1 * L, 5001)
    xm = mesh.Rectangular3D(x, [0.25 * L], [0.104])

    axhline(0., lw=0.7, color='k')
    plot(x, test.j(array(xm)), color='C0', label='current')
    xlabel(f"${config.axes[1]}$ [µm]")
    ylabel("$j$ [kA/cm]")
    legend(loc='upper left')
    twinx()

    test.solver.inCurrentDensity = CurrentDensityProvider3D(lambda m, _: array([zeros(len(m)), zeros(len(m)), test.j(array(m))]).T)
    test.solver.compute()
    plot_profile(test.solver.outCarriersConcentration(xm), color='C2', label='concentration (numeric)')

    plot(x, test.n(array(xm)), 'C1', ls='--', label='concentration (analytic)')

    xlabel(f"${config.axes[1]}$ [µm]")
    ylabel("$n$ [cm$^{-3}$]")
    legend(loc='upper right')

    show()
