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
from optical import effective


@material.simple()
class Glass(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.3


@material.simple()
class Cladding(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.28


class EffectiveIndex(unittest.TestCase):

    def setUp(self):
        self.solver = effective.EffectiveIndex2D("eim")
        rect = geometry.Rectangle(0.75, 0.5, Glass())
        space = geometry.Cartesian2D(rect, left="mirror")
        self.solver.geometry = space

    def testBasic(self):
        self.assertEqual(self.solver.id, "eim:optical.EffectiveIndex2D")

    def testExceptions(self):
        with self.assertRaisesRegex(ValueError, r"^Effective index \[0\] cannot be provided now$"):
            self.solver.outNeff(0)
        with self.assertRaisesRegex(ValueError, r"^Optical field magnitude \[0\] cannot be provided now$"):
            self.solver.outLightMagnitude(0, mesh.Rectangular2D([1,2], [3,4]))

    def testComputations(self):
        self.solver.wavelength = 1000.
        self.solver.polarization = "TE"
        self.assertAlmostEqual(self.solver.modes[self.solver.find_mode(1.15, '+')].neff, 1.1465, 3)
        self.solver.root.method = 'muller'
        self.solver.stripe_root.method = 'muller'
        self.solver.polarization = "TM"
        self.assertAlmostEqual( self.solver.modes[self.solver.find_mode(1.11, '+')].neff, 1.111, 3)

    def testMesh(self):
        mesh = self.solver.mesh

    def testRefractiveIndex(self):
        self.solver.set_simple_mesh()
        msh = self.solver.mesh.elements.mesh
        geo = self.solver.geometry
        refr = [geo.get_material(point).Nr(1000., 300.) for point in msh]
        self.assertEqual(list(self.solver.outRefractiveIndex(msh)), refr)

    def testDeltaNeffs(self):
        xx = array([0.0, 0.5, 1.0])
        neffs = self.solver.get_delta_neff(xx)
        self.assertAlmostEqual(neffs[0], 1.1829, 3)
        self.assertAlmostEqual(neffs[1], 1.1829, 3)
        self.assertAlmostEqual(neffs[2], 0.9240, 3)


class EffectiveIndexLaser(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'xy'
        rect1 = geometry.Rectangle(0.75, 0.24, Glass())
        self.rect2 = geometry.Rectangle(0.75, 0.02, Glass())
        self.rect2.role = 'gain'
        stack = geometry.Stack2D()
        stack.prepend(rect1)
        stack.prepend(self.rect2)
        stack.prepend(rect1)
        space = geometry.Cartesian2D(stack, left="mirror", length=1000)
        self.solver = effective.EffectiveIndex2D("eim")
        self.solver.geometry = space
        self.solver.mirrors = 0.7, 1.0
        self.profile = StepProfile(space)
        self.solver.inGain = self.profile.outGain

    def testThreshold(self):
        try:
            from scipy.optimize import brentq
        except ImportError:
            pass
        else:
            self.solver.stripe_root.method = 'muller'
            def fun(g):
                self.profile[self.rect2] = g
                m = self.solver.find_mode(1.15)
                return self.solver.modes[m].neff.imag
            gain = brentq(fun, 0., 100.)
            self.assertAlmostEqual(gain, 81.648, 2)

    def testAbsorptionIntegral(self):
       self.profile[self.rect2] = 81.649513489
       m = self.solver.find_mode(1.15)
       self.solver.modes[m].power = 1.4
       self.assertAlmostEqual(self.solver.get_total_absorption(m), -2.0, 1)

    def testAbsorbedHeat(self):
        self.profile[self.rect2] = 81.649513489
        m = self.solver.find_mode(1.15)
        self.solver.modes[m].power = 0.7
        box = self.solver.geometry.item.bbox
        msh = mesh.Rectangular2D(mesh.Regular(box.lower.x, box.upper.x, 1000), mesh.Regular(box.lower.y, box.upper.y, 1000))
        heat = self.solver.outHeat(msh)
        # 1e-15: µm³->m³ W->mW
        integral = 2e-15 * sum(heat) * (msh.axis0[1] - msh.axis0[0]) * (msh.axis1[1] - msh.axis1[0]) * self.solver.geometry.extrusion.length
        self.assertAlmostEqual(integral, self.solver.get_total_absorption(m), 2)
