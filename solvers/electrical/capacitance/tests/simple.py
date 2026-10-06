# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2026 Lodz University of Technology
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

from plask import *
from plask import material, geometry, mesh
from electrical.capacitance import Capacitance2D

from plask.phys import eps0  # 8.854187817e-6 pF/µm


@material.simple()
class Capacitor(material.Material):

    def cond(self, T):
        return 100.

    def eps(self, T):
        return 9.


class ConductivityMixin:

    def conductivity(self, mesh, interp=None):
        matfield = MaterialField(self.solver.geometry, mesh)
        return matfield.cond(300)


class OneLayer(ConductivityMixin, unittest.TestCase):

    def setUp(self):
        layer = geometry.Rectangle(1000., 100., Capacitor())
        layer.role = 'active'
        self.solver = Capacitance2D("capacitance2d")
        self.solver.geometry = geometry.Cartesian2D(layer, length=1000.)
        generator = mesh.Rectangular2D.DivideGenerator()
        generator.prediv = 1, 10
        self.solver.mesh = generator
        self.solver.voltage_boundary.append(self.solver.mesh.Top(), 0.)
        self.solver.voltage_boundary.append(self.solver.mesh.Bottom(), 1.)
        self.solver.frequency = 1e3
        self.solver.inDifferentialConductivity = flow.ConductivityProvider2D(self.conductivity)

    def testCurrent(self):
        self.solver.compute()
        U = self.solver.outAcVoltage(self.solver.mesh)
        correct_current = 1000. # mA
        correct_imag_current = 2 * pi * self.solver.frequency * 9. * eps0 * 1000.**2 / 100. # mA
        I = self.solver.get_ac_current()
        self.assertAlmostEqual(I.real, correct_current, 6)
        self.assertAlmostEqual(I.imag, correct_imag_current, 6)
        Iact = self.solver.get_ac_current(active=True)
        self.assertAlmostEqual(Iact, correct_current, 6)


if __name__ == '__main__':
    test = unittest.main(exit=False)
    show()
    sys.exit(not test.result.wasSuccessful())
