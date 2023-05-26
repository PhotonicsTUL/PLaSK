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


class EffectiveFrequency(unittest.TestCase):

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
                <block dr="10" dz="0.06949" material="GaAs"/>
                <stack name="top-dbr" repeat="24">
                  <block dr="10" dz="0.07955" material="AlGaAs"/>
                  <block dr="10" dz="0.06949" material="GaAs"/>
                </stack>
                <block name="x1" dr="10" dz="0.06371" material="AlGaAs"/>
                <shelf name="oxide-layer">
                  <block dr="4" dz="0.01593" material="AlAs"/><block dr="6" dz="0.01593" material="AlOx"/>
                </shelf>
                <block name="x" dr="10" dz="0.00000" material="AlGaAs"/>
                <block dr="10" dz="0.13649" material="GaAs"/>
                <shelf name="QW">
                  <block name="active" role="gain" dr="4" dz="0.00500" material="InGaAs"/><block dr="6" dz="0.00500" material="InGaAs"/>
                </shelf>
                <zero/>
                <block dr="10" dz="0.13649" material="GaAs"/>
                <stack name="bottom-dbr" repeat="29">
                  <block dr="10" dz="0.07955" material="AlGaAs"/>
                  <block dr="10" dz="0.06949" material="GaAs"/>
                </stack>
                <block dr="10" dz="0.07955" material="AlGaAs"/>
                </stack>
              </cylindrical>
            </geometry>
            <solvers>
              <optical name="efm" lib="effective" solver="EffectiveFreqCyl">
                <geometry ref="vcsel"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solvers.efm
        self.solver.lam0 = 980.
        self.profile = StepProfile(self.solver.geometry)
        self.solver.inGain = self.profile.outGain

    def testField(self):
        axis0 = plask.mesh.Regular(0.001, 39.999, 100000)
        #axis0 = linspace(0.01, 3.99, 200)
        #axis1  = [ self.manager.geo.layers.bbox.lower.z-1e-6, 0.0025, self.manager.geo.layers.bbox.upper.z+-1e-6 ]
        axis1  = plask.mesh.Ordered([self.manager.geo.layers.bbox.upper.z+-1e-6])
        dr = axis0[1]-axis0[0]
        msh = mesh.Rectangular2D(axis0, axis1)
        self.solver.find_mode(980.1, 0)
        self.solver.modes[0].power = 2000.
        field = self.solver.outLightMagnitude(0,msh).array[:,-1]
        integral = 2e-12 * pi * sum(field * msh.axis0) * dr
        self.assertAlmostEqual(integral, 2., 4)

    def testRefractiveIndex(self):
        self.solver.set_simple_mesh()
        msh = self.solver.mesh.elements.mesh
        geo = self.solver.geometry
        refr = [geo.get_material(point).Nr(980., 300.) for point in msh]
        self.assertEqual(
            ["%.8g" % nr[0].real for nr in self.solver.outRefractiveIndex(msh)],
            ["%.8g" % r.real for r in refr]
        )

    def testComputations(self):
        m = self.solver.find_mode(980.1)
        self.assertEqual(m, 0)
        self.assertEqual(len(self.solver.modes), 1)
        self.assertAlmostEqual(self.solver.modes[m].lam, 979.702-0.021j, 3)

    def testThreshold(self):
        try:
            from scipy.optimize import brentq
        except ImportError:
            pass
        else:
            def fun(g):
                self.profile[self.manager.geo['active']] = g
                m = self.solver.find_mode(980.1)
                return imag(self.solver.modes[m].lam)
            threshold = brentq(fun, 0., 2000., xtol=1e-6)
            self.assertAlmostEqual(threshold, 1181.7, 1)

    def testAbsorptionIntegral(self):
        self.profile[self.manager.geo['active']] = 1181.6834
        m = self.solver.find_mode(980.1)
        self.solver.modes[m].power = 2.0
        box = self.solver.geometry.item.bbox
        field = self.solver.outLightMagnitude(m, mesh.Rectangular2D(mesh.Ordered([0.]), mesh.Ordered([box.lower.z, box.upper.z])))
        total_power = self.solver.modes[m].power * (1. + 3.53 * field[0] / field[1])
        self.assertAlmostEqual(-self.solver.get_total_absorption(m), total_power, 2)

    def testAbsorbedHeat(self):
        self.profile[self.manager.geo['active']] = 1181.6834
        m = self.solver.find_mode(980.1)
        self.solver.modes[m].power = 2.0
        box = self.solver.geometry.get_object_bboxes(self.manager.geo['active'])[0]
        msh = mesh.Rectangular2D(mesh.Regular(0., 10., 1000), [0.5 * (box.lower.z + box.upper.z)])
        heat = self.solver.outHeat(msh).array[:,0]
        # 1e-15: µm³->m³ W->mW
        integral = 2e-15 * pi * sum(heat * msh.axis0) * (box.upper.z - box.lower.z) * (msh.axis0[1] - msh.axis0[0])
        self.assertAlmostEqual(integral, self.solver.get_total_absorption(m), 2)

    def testNeffs(self):
        rr = array([0.0, 2.0, 6.0])
        neffs = self.solver.get_nng(rr)
        self.assertAlmostEqual(neffs[0], 3.3174, 3)
        self.assertAlmostEqual(neffs[1], 3.3174, 3)
        self.assertAlmostEqual(neffs[2], 3.2816, 3)

    def testDeltaNeffs(self):
        rr = array([0.0, 2.0, 6.0])
        neffs = self.solver.get_delta_neff(rr)
        self.assertAlmostEqual(neffs[0], 0.0244635-0.0095423j, 6)
        self.assertAlmostEqual(neffs[1], 0.0244635-0.0095423j, 6)
        self.assertAlmostEqual(neffs[2], 0.0006568-0.3719099j, 6)
