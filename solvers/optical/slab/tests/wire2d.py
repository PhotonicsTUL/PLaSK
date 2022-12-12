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
from optical import slab

config.axes = 'zxy'

@material.simple()
class Glass(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.30


@material.simple()
class Cladding(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.28


class Wire(unittest.TestCase):

    def setUp(self):
        self.solver = slab.Fourier2D("fourier")
        wire_stack = geometry.Stack2D()
        wire_stack.append(geometry.Rectangle(0.75, 0.125, Glass()))
        rect = geometry.Rectangle(0.75, 0.375, Glass())
        rect.role = 'interface'
        wire_stack.append(rect)
        space = geometry.Cartesian2D(wire_stack, left="mirror")
        self.solver.geometry = space
        self.solver.lam = 1000.

    def testComputations(self):
        self.solver.symmetry = 'Etran'
        # self.solver.polarization = "Etran"
        self.assertAlmostEqual(self.solver.modes[self.solver.find_mode(neff=1.15)].neff, 1.147, 3)
        self.solver.symmetry = 'Htran'
        # self.solver.polarization = "Htran"
        self.assertAlmostEqual( self.solver.modes[self.solver.find_mode(neff=1.05)].neff, 1.054, 3)

    def testFullDeterminant(self):
        self.solver.determinant_type = 'full'
        self.solver.symmetry = 'Etran'
        # self.solver.polarization = "Etran"
        self.assertAlmostEqual(self.solver.modes[self.solver.find_mode(neff=1.15)].neff, 1.147, 3)
        self.solver.symmetry = 'Htran'
        # self.solver.polarization = "Htran"
        self.assertAlmostEqual( self.solver.modes[self.solver.find_mode(neff=1.05)].neff, 1.054, 3)


# class WireLaser(unittest.TestCase):
#
#     def setUp(self):
#         plask.config.axes = 'xy'
#         rect1 = geometry.Rectangle(0.75, 0.24, Glass())
#         self.rect2 = geometry.Rectangle(0.75, 0.02, Glass())
#         self.rect2.role = 'gain'
#         stack = geometry.Stack2D()
#         stack.prepend(rect1)
#         stack.prepend(self.rect2)
#         stack.prepend(rect1)
#         space = geometry.Cartesian2D(stack, left="mirror", length=1000)
#         self.solver = slab.Fourier2D("fourier")
#         self.solver.geometry = space
#         self.solver.mirrors = 0.7, 1.0
#         self.profile = StepProfile(space)
#         self.solver.inGain = self.profile.outGain
#
#     def testThreshold(self):
#         try:
#             from scipy.optimize import brentq
#         except ImportError:
#             pass
#         else:
#             self.solver.stripe_root.method = 'muller'
#             def fun(g):
#                 self.profile[self.rect2] = g
#                 m = self.solver.find_mode(1.15)
#                 return self.solver.modes[m].neff.imag
#             gain = brentq(fun, 0., 100.)
#             self.assertAlmostEqual(gain, 81.648, 2)
#
#     def testAbsorptionIntegral(self):
#        self.profile[self.rect2] = 81.649513489
#        m = self.solver.find_mode(1.15)
#        self.solver.modes[m].power = 1.4
#        self.assertAlmostEqual(self.solver.get_total_absorption(m), -2.0, 1)
#
#     def testAbsorbedHeat(self):
#         self.profile[self.rect2] = 81.649513489
#         m = self.solver.find_mode(1.15)
#         self.solver.modes[m].power = 0.7
#         box = self.solver.geometry.item.bbox
#         msh = mesh.Rectangular2D(mesh.Regular(box.lower.x, box.upper.x, 1000), mesh.Regular(box.lower.y, box.upper.y, 1000))
#         heat = self.solver.outHeat(msh)
#         # 1e-15: µm³->m³ W->mW
#         integral = 2e-15 * sum(heat) * (msh.axis0[1] - msh.axis0[0]) * (msh.axis1[1] - msh.axis1[0]) * self.solver.geometry.extrusion.length
#         self.assertAlmostEqual(integral, self.solver.get_total_absorption(m), 2)


if __name__ == '__main__':
    wire = Wire()
    wire.setUp()
    neffs = linspace(1.0, 1.3, 301)
    wire.solver.symmetry = 'Etran'
    det1 = wire.solver.get_determinant(neff=neffs)
    wire.solver.determinant_type = 'full'
    det2 = wire.solver.get_determinant(neff=neffs)
    plot(neffs, abs(det1))
    plot(neffs, abs(det2))
    xlabel("$n_\\mathrm{eff}$")
    ylabel("Determinant")
    yscale('log')
    show()
