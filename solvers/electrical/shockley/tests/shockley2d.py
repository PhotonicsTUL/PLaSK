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
from electrical.shockley import Shockley2D, ShockleyCyl

eps0 = 8.854187817e-6 # pF/µm

@material.simple()
class Conductor(material.Material):
    def cond(self, T):
        return (1e+9, 1e+9)
    def eps(self, T):
        return 1.


class Shockley2D_Test(unittest.TestCase):

    def setUp(self):
        rect = geometry.Rectangle(1000., 300., Conductor())
        junc = geometry.Rectangle(1000., 0.02, 'GaAs')
        junc.role = 'active'
        stack = geometry.Stack2D()
        stack.append(rect)
        stack.append(junc)
        stack.append(rect)
        space = geometry.Cartesian2D(stack, length=1000.)
        self.solver = Shockley2D("electrical2d")
        self.solver.geometry = space
        generator = mesh.Rectangular2D.DivideGenerator()
        generator.prediv = 2,1
        self.solver.mesh = generator
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.maxerr = 1e-5
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)

    def testComputations(self):
        self.solver.compute()
        correct_current = 1e-3 * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual(self.solver.get_total_current(), correct_current, 3)
        capacitance = eps0 * material.GaAs().eps() * 1000.**2 / 0.02 # pF
        self.assertAlmostEqual(self.solver.get_capacitance(), capacitance, 2)
        heat = correct_current * 1.
        self.assertAlmostEqual(self.solver.get_total_heat(), heat, 3)


    def testConductivity(self):
        msh = self.solver.mesh.elements.mesh
        geo = self.solver.geometry
        conds = [geo.get_material(point).cond(300.) if not geo.has_role('active', point) else (0.,5.) for point in msh]
        self.assertSequenceEqual(self.solver.outConductivity(msh), conds)



class ShockleyCyl_Test(unittest.TestCase):

    def setUp(self):
        cont = geometry.Rectangle(400., 100., Conductor())
        rect = geometry.Rectangle(1000., 300., Conductor())
        junc = geometry.Rectangle(1000., 0.02, "GaAs")
        junc.role = 'active'
        shelf = geometry.Shelf()
        shelf.append(cont)
        shelf.append_gap(200.)
        shelf.append(cont)
        stack = geometry.Stack2D()
        stack.prepend(shelf)
        stack.prepend(rect)
        stack.prepend(junc)
        stack.prepend(rect)
        space = geometry.Cylindrical(stack)
        self.solver = ShockleyCyl("electricalcyl")
        self.solver.geometry = space
        generator = mesh.Rectangular2D.DivideGenerator()
        generator.prediv = 2,1
        self.solver.mesh = generator
        self.solver.beta = 10.
        self.solver.js = 1.
        self.solver.maxerr = 1e-5
        self.solver.voltage_boundary.append(self.solver.mesh.TopOf(cont), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)

    def testComputations(self):
        self.solver.compute(1000)
        correct_current = 1e-3 * pi * self.solver.js * (exp(self.solver.beta) - 1)
        self.assertAlmostEqual(self.solver.get_total_current(), correct_current, 3)
        capacitance = eps0 * material.GaAs().eps() * pi*1000.**2 / 0.02 # pF
        self.assertAlmostEqual(self.solver.get_capacitance(), capacitance, 2)
        heat = correct_current * 1.
        self.assertAlmostEqual(self.solver.get_total_heat(), heat, 3)

    def testComputationsTemp(self):
        self.solver.beta = lambda T: log(T * 70)
        self.solver.compute(1000)
        correct_current = 1e-3 * pi * self.solver.js * (21000 - 1)
        self.assertAlmostEqual(self.solver.get_total_current(), correct_current, 3)
        self.solver.inTemperature = 250
        self.solver.compute(1000)
        correct_current = 1e-3 * pi * self.solver.js * (17500 - 1)
        self.assertAlmostEqual(self.solver.get_total_current(), correct_current, 3)

    def testConductivity(self):
        msh = self.solver.mesh.elements.mesh
        geo = self.solver.geometry
        reference = [geo.get_material(point).cond(300.) if not geo.has_role('active', point) else (0.,5.) for point in msh]
        ac = material.Air().cond(300.)
        result = [ac if isnan(c[0]) else c for c in self.solver.outConductivity(msh)]
        self.assertListEqual(result, reference)

    if __name__ == '__main__':
        def testGeometry(self):
            fig = figure()
            plot_geometry(self.solver.geometry, margin=0.1, color='k')
            plot_mesh(self.solver.mesh, color='c')
            plot_boundary(self.solver.voltage_boundary, self.solver.mesh, self.solver.geometry, cmap='bwr', s=40)


if __name__ == '__main__':
    test = unittest.main(exit=False)
    show()
    sys.exit(not test.result.wasSuccessful())
