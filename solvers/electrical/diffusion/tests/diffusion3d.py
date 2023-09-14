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

from electrical.diffusion import Diffusion3D

config.loglevel = 'debug'
config.axes = 'xyz'

@material.simple('GaAs')
class GaAsQW(material.Material):

    def A(self, T=300):
        return 3e7

    def B(self, T=300):
        return 1.7e-10

    def C(self, T=300):
        return 6e-30

    def D(self, T=300):
        return 10.


class DiffusionTest(unittest.TestCase):

    def setUp(self):
        # clad = geometry.Cuboid(0.500, 0.500, 0.100, 'GaAs')
        # qw = geometry.Cuboid(0.500, 0.500, 0.002, 'GaAsQW')
        # ba = geometry.Cuboid(0.500, 0.500, 0.001, 'GaAs')
        clad = geometry.Cylinder(0.500, 0.100, 'GaAs')
        qw = geometry.Cylinder(0.500, 0.002, 'GaAsQW')
        ba = geometry.Cylinder(0.500, 0.001, 'GaAs')
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
        self.solver.geometry = self.geometry
        self.solver.maxerr = 0.0001
        self.qwm = qw.material
        self.test_mesh = mesh.Rectangular3D(linspace(-0.4, 0.4, 5), linspace(-0.4, 0.4, 5), [0.104])

    def test_uniform(self):
        n = 1.0e19
        j = 1e-7 * (self.qwm.A() * n + self.qwm.B() * n**2 + self.qwm.C() * n**3) * (phys.qe * 0.006)
        # self.solver.mesh = mesh.Rectangular3D.SimpleGenerator()
        self.solver.inCurrentDensity = vec(0., 0., j)
        self.solver.compute()
        res = self.solver.outCarriersConcentration(self.test_mesh).array[:,:,0]
        ref = n * ones_like(res)
        y, x = meshgrid(self.test_mesh.axis0, self.test_mesh.axis1)
        mask = x**2 + y**2 <= 0.5**2
        ref[~mask] = 0.
        assert_allclose(res, ref)


if __name__ == '__main__':
    unittest.main()
