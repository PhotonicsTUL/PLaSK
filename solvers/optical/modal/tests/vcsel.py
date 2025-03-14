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

import sys

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import modal


plask.config.axes = "rz"


class VCSEL(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'rz'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask loglevel="debug">
            <materials>
              <material name="GaAs" base="semiconductor">
                <nr>3.53</nr>
                <absp>0.</absp>
              </material>
              <material name="AlGaAs" base="semiconductor">
                <nr>3.08</nr>
                <absp>0.</absp>
              </material>
              <material name="AlAs" base="semiconductor">
                <nr>2.95</nr>
                <absp>0.</absp>
              </material>
              <material name="AlOx" base="semiconductor">
                <nr>1.53</nr>
                <absp>0.</absp>
              </material>
              <material name="InGaAs" base="semiconductor">
                <nr>3.53</nr>
                <absp>0.</absp>
              </material>
            </materials>
            <geometry>
              <cylindrical axes="rz" name="vcsel" outer="extend" bottom="GaAs">
                <stack name="layers">
                <block dr="5." dz="0.06949" material="GaAs"/>
                <stack name="top-dbr" repeat="24">
                  <block dr="5." dz="0.07955" material="AlGaAs"/>
                  <block dr="5." dz="0.06949" material="GaAs"/>
                </stack>
                <item zero="0">
                  <stack name="middle">
                    <block name="x1" dr="5." dz="0.06371" material="AlGaAs"/>
                    <shelf name="oxide-layer">
                      <block dr="4." dz="0.01593" material="AlAs"/><block dr="1." dz="0.01593" material="AlOx"/>
                    </shelf>
                    <block name="x" dr="5." dz="0.00000" material="AlGaAs"/>
                    <block dr="5." dz="0.13649" material="GaAs"/>
                    <shelf name="QW">
                      <block name="active" role="gain" dr="4." dz="0.00500" material="InGaAs"/><block dr="1." dz="0.00500" material="InGaAs"/>
                    </shelf>
                    <zero/>
                    <block dr="5." dz="0.13649" material="GaAs"/>
                  </stack>
                </item>
                <stack name="bottom-dbr" repeat="29">
                  <block dr="5." dz="0.07955" material="AlGaAs"/>
                  <block dr="5." dz="0.06949" material="GaAs"/>
                </stack>
                <block dr="5." dz="0.07955" material="AlGaAs"/>
                </stack>
              </cylindrical>
            </geometry>
            <solvers>
              <optical name="bessel" lib="modal" solver="BesselCyl">
                <geometry ref="vcsel"/>
                <expansion domain="finite" lam0="980." k-scale="0.1" k-method="laguerre"/>
                <mode emission="top"/>
                <pml dist="20." factor="1-0j" size="2.0"/>
                <interface object="QW"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solvers.bessel
        self.solver.initialize()
        self.profile = StepProfile(self.solver.geometry)
        self.solver.inGain = self.profile.outGain
        self.solver.lam0 = 980.
        self.solver.size = 30

    def testComputationsFinite(self):
        self.solver.domain = 'finite'
        m = self.solver.find_mode(980.1)
        self.assertEqual(m, 0)
        self.assertEqual(len(self.solver.modes), 1)
        # self.assertAlmostEqual(self.solver.modes[m].lam.real,  979.59, 2)
        self.assertAlmostEqual(self.solver.modes[m].lam.real,  979.587, 3)
        self.assertAlmostEqual(self.solver.modes[m].lam.imag, -0.02077, 3)

        # Test integration of the Pointing vector
        R = 27.
        n = 1000
        dr = 1e-6*R / n
        rr = linspace(0., R, n+1)
        msh = mesh.Rectangular2D(rr, [self.solver.geometry.bbox.top + 1e-6])
        E = self.solver.outLightE(m, msh).array[:,0,:]
        H = self.solver.outLightH(m, msh).array[:,0,:]
        P = 0.5 * real(E[:,1]*conj(H[:,0]) - E[:,0]*conj(H[:,1]))
        self.assertAlmostEqual(2e3*pi * sum(1e-6*rr * P) * dr / self.solver.modes[m].power, 1.0, 3)

    def testComputationsInfinite(self):
        self.solver.domain = 'infinite'
        self.solver.kscale = 0.05
        m = self.solver.find_mode(979.0)
        self.assertEqual(m, 0)
        self.assertEqual(len(self.solver.modes), 1)
        self.assertAlmostEqual(self.solver.modes[m].lam.real,  979.663, 3)
        self.assertAlmostEqual(self.solver.modes[m].lam.imag, -0.02077, 3)

        # Test integration of the Pointing vector
        R = 27.
        n = 1000
        dr = 1e-6*R / n
        rr = linspace(0., R, n+1)
        msh = mesh.Rectangular2D(rr, [self.solver.geometry.bbox.top + 1e-6])
        E = self.solver.outLightE(m, msh).array[:,0,:]
        H = self.solver.outLightH(m, msh).array[:,0,:]
        P = 0.5 * real(E[:,1]*conj(H[:,0]) - E[:,0]*conj(H[:,1]))
        self.assertAlmostEqual(2e3*pi * sum(1e-6*rr * P) * dr / self.solver.modes[m].power, 1.0, 2)

    def _integrals_test(self, domain, prec, right=None, nr=101, kscale=None):
        self.solver.domain = domain
        if kscale is not None:
            self.solver.kscale = kscale
        bbox = self.solver.geometry.get_object_bboxes(self.manager.geo.middle)[0]
        if right is None:
            right = bbox.right
        msh = mesh.Rectangular2D(mesh.Regular(0., right, nr), mesh.Regular(bbox.bottom, bbox.top, 501))
        m = self.solver.find_mode(979.0)
        dr = msh.axis0[1] - msh.axis0[0]
        dz = msh.axis1[1] - msh.axis1[0]
        integral_mesh = msh.elements.mesh
        rr, _ = meshgrid(integral_mesh.axis0, integral_mesh.axis1, indexing='ij')

        E = self.solver.outLightE(integral_mesh).array
        E2 = sum(real(E*conj(E)), -1)
        EE0 = 0.5 * 2*pi * sum((rr * E2).ravel()) * dr * dz
        EE1 = self.solver.integrateEE(bbox.bottom, bbox.top)
        self.assertAlmostEqual(EE0 / EE1, 1., prec)

        H = self.solver.outLightH(integral_mesh).array
        H2 = sum(real(H*conj(H)), -1)
        HH0 = 0.5 * 2*pi * sum((rr * H2).ravel()) * dr * dz
        HH1 = self.solver.integrateHH(bbox.bottom, bbox.top)
        self.assertAlmostEqual(HH0 / HH1, 1., prec)

    def testIntegralsFinite(self):
        self._integrals_test('finite', 2, kscale=0.1)

    def testIntegralsInfinite(self):
        self._integrals_test('infinite', 2, 20., 301, kscale=0.05)

    def plot_determinant(self):
        lams = linspace(979., 982., 201)
        dets = self.solver.get_determinant(lam=lams, m=1)
        plot(lams, abs(dets))
        yscale('log')

    def plot_field(self):
        self.solver.find_mode(979.0, m=1)
        print_log('result', self.solver.modes[0])
        box = self.solver.geometry.bbox
        msh = mesh.Rectangular2D(mesh.Regular(0., box.right, 101),
                                 mesh.Regular(box.bottom, box.top, 1001))
        field = self.solver.outLightE(msh)
        mag = max(abs(field.array.ravel()))
        scale = linspace(-mag, mag, 255)
        figure()
        plot_geometry(self.solver.geometry, color='k', alpha=0.15)
        plot_field(field, scale, comp='r', cmap='bwr')
        window_title("Er")
        colorbar(use_gridspec=True)
        tight_layout(pad=0.1)
        figure()
        plot_geometry(self.solver.geometry, color='k', alpha=0.15)
        plot_field(field, scale, comp='p', cmap='bwr')
        colorbar(use_gridspec=True)
        window_title("Ep")
        tight_layout(pad=0.1)
        figure()
        plot_geometry(self.solver.geometry, color='k', alpha=0.15)
        plot_field(field, scale, comp='z', cmap='bwr')
        colorbar(use_gridspec=True)
        window_title("Ez")
        tight_layout(pad=0.1)

        figure()
        plot_geometry(self.solver.geometry, color='w', alpha=0.15)
        light = self.solver.outLightMagnitude(msh)
        plot_field(light)
        colorbar(use_gridspec=True)
        window_title("Mag")
        tight_layout(pad=0.1)

        z = self.solver.geometry.get_object_bboxes(self.manager.geo.QW)[0].center.z
        arr = light.array
        r = msh.axis0[int(unravel_index(argmax(arr), arr.shape)[0])]
        rmsh = mesh.Rectangular2D(linspace(0, box.right, 2001), [z])
        zmsh = mesh.Rectangular2D([r], linspace(box.bottom, box.top, 10001))
        figure()
        plot_profile(self.solver.outLightMagnitude(rmsh))
        window_title(u"Horizontal (z = {:.1f} µm".format(z))
        tight_layout(pad=0.1)
        figure()
        plot_profile(self.solver.outLightMagnitude(zmsh), swap_axes=True)
        window_title(u"Vertical (r = {:.1f} µm".format(r))
        tight_layout(pad=0.1)


if __name__ == "__main__":
    vcsel = VCSEL('plot_field')
    vcsel.setUp()

    try:
        vcsel.solver.domain = sys.argv[1]
    except IndexError:
        pass

    # vcsel.plot_determinant()
    vcsel.plot_field()
    show()
