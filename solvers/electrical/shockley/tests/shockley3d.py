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
from electrical.shockley import Shockley3D

eps0 = 8.854187817e-6 # pF/µm

config.axes = 'xyz'

@material.simple()
class Conductor(material.Material):
    def cond(self, T):
        return (1e+9, 1e+9)
    def eps(self, T):
        return 1.


class Shockley3D_Test(unittest.TestCase):

    def setUp(self):
        cond = Conductor()
        contact = geometry.Cuboid(700., 700., 1., cond)
        #rect = geometry.Cylinder(500., 300., cond)
        #junc = geometry.Cylinder(500., 0.02, 'GaAs')
        #self.S = pi * 0.25e6
        rect = geometry.Cuboid(1000., 1000., 300., cond)
        junc = geometry.Cuboid(1000., 1000., 0.02, 'GaAs')
        self.S = 1e6
        junc.role = 'active'
        stack = geometry.Stack3D(xcenter=0,  ycenter=0)
        top = stack.prepend(contact)
        stack.prepend(rect)
        stack.prepend(junc)
        stack.prepend(rect)
        bottom = stack.prepend(contact)
        space = geometry.Cartesian3D(stack)
        self.solver = Shockley3D("electrical3d")
        self.solver.geometry = space
        self.solver.mesh = mesh.Rectangular3D.DivideGenerator(prediv=(3,3,2), gradual=False)
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.maxerr = 1e-3
        self.solver.voltage_boundary.append(self.solver.mesh.TopOf(contact,top), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.BottomOf(contact, bottom), 1.)

    def testComputations(self):
        self.solver.compute(1000)
        correct_current = 1e-9 * self.S * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual(self.solver.get_total_current(), correct_current, 3)
        capacitance = eps0 * material.GaAs().eps() * self.S / 0.02 # pF
        self.assertAlmostEqual(self.solver.get_capacitance(), capacitance, 2)
        heat = correct_current * 1.
        self.assertAlmostEqual(self.solver.get_total_heat(), heat, 3)

    def testComputationsExcluded(self):
        self.solver.empty_elements = 'exclude'
        self.testComputations()

    def testComputationsTemp(self):
        self.solver.beta = lambda T: log(T * 70)
        self.solver.compute(1000)
        correct_current = 1e-9 * self.S * self.solver.js * (21000 - 1)
        self.assertAlmostEqual(self.solver.get_total_current(), correct_current, 3)
        self.solver.inTemperature = 250
        self.solver.compute(1000)
        correct_current = 1e-9 * self.S * self.solver.js * (17500 - 1)
        self.assertAlmostEqual(self.solver.get_total_current(), correct_current, 3)

    def testConductivity(self):
        msh = self.solver.mesh.elements.mesh
        geo = self.solver.geometry
        conds = [geo.get_material(point).cond(300.) if not geo.has_role('active', point) else (0.,5.) for point in msh]
        ac = material.Air().cond(300.)
        result = [ac if isnan(c[0]) else c for c in self.solver.outConductivity(msh)]
        self.assertSequenceEqual(result, conds)

    if __name__ == '__main__':
        def testGeometry(self):
            for plane in ('xy', 'xz', 'yz'):
                fig = figure()
                plot_geometry(self.solver.geometry, plane=plane, margin=0.1, color='k')
                plot_mesh(self.solver.mesh, plane=plane, color='c')
                plot_boundary(self.solver.voltage_boundary, self.solver.mesh, self.solver.geometry, plane=plane,
                              cmap='bwr', s=40)
                window_title(plane)


if __name__ == '__main__':
    test = unittest.main(exit=False)
    show()
    sys.exit(not test.result.wasSuccessful())
