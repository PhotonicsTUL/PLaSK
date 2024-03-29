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
from plask import material, geometry, mesh, flow

from gain import FreeCarrierCyl

plask.config.axes = 'rz'

@material.simple('GaAs')
class Barrier(material.Material):
    lattC = 5.654
    Dso = 0.3548
    Me = 0.103
    Mhh = 0.6
    Mlh = 0.14
    D = 10
    def VB(self, T=300., e=0., point=None, hole=None):
        return -0.75
    def A(self, T=300.):
        return 7e7 + (T-300) * 1.4e5
    def B(self, T=300.):
        return 1.1e-10 - (T-300) * -2.2e-13
    def C(self, T=300.):
        return 1.130976e-28 - (T-300) * 1.028743e-30 + (T-300)**2 * 8.694142e-32
    def Eg(self, T=300., e=0., point=None):
        return 1.30 - 0.0003 * (T-300)
    def nr(self, wl, T=300., n=0.):
        return 3.3 + (wl-1310) * 5.00e-4 + (T-300) * 7e-4
    def absp(self, wl, T=300.):
        return 50. + (T-300) * 7e-2


@material.simple('Barrier')
class Well(material.Material):
    Me = 0.052
    Mhh = 0.477
    Mlh = 0.103
    absp = 1000.
    def VB(self, T=300., e=0., point=None, hole=None):
        return 0.14676 + 0.000033 * (T-300) - 0.75
    def Eg(self, T=300., e=0., point=None):
        return 0.87 - 0.0002 * (T-300) - 100. * e
    def nr(self, wl, T=300., n=0.):
        return 3.6 + (wl-1310) * 5.00e-4 + (T-300) * 7e-4


@material.simple('Barrier')
class Barrier0(material.Material):
    def VB(self, T=300., e=0., point=None, hole=None):
        return -0.5
    def Eg(self, T=300., e=0., point=None):
        return 0.9


@material.simple('Well')
class Well0(material.Material):
    def VB(self, T=300., e=0., point=None, hole=None):
        return -0.2
    def Eg(self, T=300., e=0., point=None):
        return 0.1


class TestStructureGain(unittest.TestCase):

    def build_geometry(self, well_material, barrier_material):
        substrate = geometry.Rectangle(20., 500., 'GaAs')
        well = geometry.Rectangle(20., 0.0060, well_material)
        well.role = 'QW'
        barrier = geometry.Rectangle(20., 0.0067, barrier_material)
        stack = geometry.Stack2D()
        cap = geometry.Rectangle(10., 500., 'GaAs')
        shelf = geometry.Shelf2D()
        shelf.append(cap)
        shelf.append(cap)
        stack.prepend(shelf)
        active = geometry.Stack2D()
        active.role = 'active'
        for i in range(4):
            active.prepend(barrier)
            active.prepend(well)
        active.prepend(barrier)
        stack.prepend(active)
        stack.prepend(substrate)
        return geometry.Cylindrical(stack), active, well

    def setUp(self):
        self.msh = mesh.Rectangular2D([0.], [500.0100])
        self.geometry, self.active, well = self.build_geometry('Well', 'Barrier')
        self.concentration = plask.StepProfile(self.geometry)
        self.concentration[well] = 3.5e18

    def get_bands(self, band, dest_mesh, interp=None):
        if band == 'CONDUCTION':
            return array([self.geometry.get_material(p).CB() for p in dest_mesh])
        elif band == 'VALENCE_LIGHT':
            return array([self.geometry.get_material(p).VB(hole='L') for p in dest_mesh])
        elif band == 'VALENCE_HEAVY':
            return array([self.geometry.get_material(p).VB(hole='H') for p in dest_mesh])
        elif band == 'SPINOFF':
            return array([self.geometry.get_material(p).Dso() for p in dest_mesh])

    def get_fermi_level(self, which, dest_mesh, interp=None):
        value = {'ELECTRONS': 0.4290875146658184, 'HOLES': -0.5893720196570111}[which]
        return array([value] * len(dest_mesh))

    def plot_bands(self):
        box = self.geometry.get_object_bboxes(self.active)[0]
        zz = linspace(box.lower.z-0.002, box.upper.z+0.002, 1001)
        msh = mesh.Rectangular2D([0.], zz)
        CC = self.get_bands('CONDUCTION', msh)
        VV = self.get_bands('VALENCE_HEAVY', msh)
        plot(1e3*zz, CC)
        plot(1e3*zz, VV)
        xlim(1e3*zz[0], 1e3*zz[-1])
        xlabel("$z$ (nm)")
        ylabel("Band Edges (eV)")
        window_title("Band Edges")
        tight_layout(pad=0.5)

    def test_gain(self):
        solver = FreeCarrierCyl("self.solver")
        solver.strained = True
        solver.substrate = 'GaAs'
        solver.geometry = self.geometry
        solver.inCarriersConcentration = self.concentration.outCarriersConcentration
        self.assertAlmostEqual(solver.outGain(self.msh, 1275.)[0][0], 1314., 0)
        msh = mesh.Rectangular2D([0.], [-400.])
        self.assertEqual(len(solver.outEnergyLevels('ELECTRONS', msh)[0]), 0)

    def test_band_edges_receiver(self):
        solver = FreeCarrierCyl("self.solver")
        geom, _, _ = self.build_geometry('Well0', 'Barrier0')
        solver.geometry = geom
        solver.inBandEdges = flow.BandEdgesProviderCyl(self.get_bands)
        solver.inCarriersConcentration = self.concentration.outCarriersConcentration
        self.assertAlmostEqual(solver.outGain(self.msh, 1275.)[0][0], 1254., 0)
        assert_allclose(
            solver.outEnergyLevels('ELECTRONS', self.msh)[0],
            [0.3337, 0.3337, 0.3337, 0.3337, 0.5259, 0.5259, 0.5263, 0.5263, 0.5979, 0.5987],
        rtol=1e-3)
        assert_allclose(
            solver.outEnergyLevels('HEAVY_HOLES', self.msh)[0],
            [-0.6166, -0.6166, -0.6166, -0.6166, -0.6561, -0.6561, -0.6562, -0.6562, -0.7174, -0.7174, -0.7174,
            -0.7174, -0.7978, -0.7916, -0.7859, -0.7813, -0.7788, -0.7651, -0.7627, -0.7606, -0.7591, -0.7586],
        rtol=1e-3)
        assert_allclose(
            solver.outEnergyLevels('LIGHT_HOLES', self.msh)[0],
            [-0.6415, -0.6415, -0.6415, -0.6415, -0.7386, -0.7386, -0.7390, -0.7390, -0.7997, -0.7844, -0.7833],
        rtol=1e-3)

    def test_fermi_levels_receiver(self):
        solver = FreeCarrierCyl("self.solver")
        solver.geometry = self.geometry
        solver.inCarriersConcentration = 0.
        solver.inFermiLevels = flow.FermiLevelsProviderCyl(self.get_fermi_level)
        self.assertAlmostEqual(solver.outGain(self.msh, 1275.)[0][0], 1254., 0)


if __name__ == '__main__':
    test = unittest.main(exit=False)
    instance = TestStructureGain('plot_bands')
    instance.setUp()
    instance.plot_bands()
    show()
    sys.exit(not test.result.wasSuccessful())
