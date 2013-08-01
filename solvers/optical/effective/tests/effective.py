#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.effective import EffectiveIndex2D, EffectiveFrequencyCyl

@material.simple
class Glass(material.Material):
    def Nr(self, wl, T): return 1.3

@material.simple
class Cladding(material.Material):
    def Nr(self, wl, T): return 1.28

Z0 = 119.9169832*pi # free space impedance

class EffectiveIndex2D_Test(unittest.TestCase):

    def setUp(self):
        self.solver = EffectiveIndex2D("eim")
        rect = geometry.Rectangle(0.75, 0.5, Glass())
        space = geometry.Cartesian2D(rect, left="mirror")
        self.solver.geometry = space

    def testBasic(self):
        self.assertEqual( self.solver.id, "eim:optical.EffectiveIndex2D" )

    def testExceptions(self):
        with self.assertRaisesRegexp(TypeError, r"^No provider nor value for wavelength$"):
            self.solver.inWavelength()
        with self.assertRaisesRegexp(ValueError, r"^Effective index cannot be provided now$"):
            self.solver.outNeff()
        with self.assertRaisesRegexp(ValueError, r"^Intensity profile cannot be provided now$"):
            self.solver.outIntensity(mesh.Rectilinear2D([1,2],[3,4]))

    def testSymmetry(self):
        self.assertIsNone( self.solver.symmetry )
        self.solver.symmetry = "-"
        self.assertEqual( self.solver.symmetry, "negative" )

    def testReceivers(self):
        self.solver.inWavelength = 850.
        self.assertEqual( self.solver.inWavelength(), 850. )

    def testComputations(self):
        self.solver.inWavelength = 1000.
        self.solver.symmetry = "+"
        self.solver.polarization = "TE"
        self.assertAlmostEqual( self.solver.compute(1.15), 1.147, 3)
        self.solver.polarization = "TM"
        self.assertAlmostEqual( self.solver.compute(1.11), 1.111, 3)

    def testMesh(self):
        mesh = self.solver.mesh

    def testField(self):
        self.solver.inWavelength = 1000.
        self.solver.symmetry = "+"
        #self.solver.symmetry = "0"
        #self.solver.set_horizontal_mesh([-0.75, 0.00, 0.75])
        self.solver.polarization = "TE"
        self.solver.compute(1.15)
        axis0 = linspace(-2.75, 2.75, 2000)
        axis1 = linspace(-2., 2.5, 1000)
        msh = mesh.Rectilinear2D(axis0, axis1).get_midpoints()
        dx = (axis0[1] - axis0[0]) * 1e-6
        dy = (axis1[1] - axis1[0]) * 1e-6
        field = self.solver.outIntensity(msh).array
        integral = sum(field) * dx*dy
        power = 1e3 * Z0 * integral # 1e3: W -> mW
        self.assertAlmostEqual(power, 1., 4)


class EffectiveFrequencyCyl_Test(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'rz'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask loglevel="debug">
            <materials>
            <material name="GaAs" kind="semiconductor">
              <nr>3.53</nr>
            </material>
            <material name="AlGaAs" kind="semiconductor">
              <nr>3.08</nr>
            </material>
            <material name="AlAs" kind="semiconductor">
              <nr>2.95</nr>
            </material>
            <material name="InGaAs" kind="semiconductor">
              <nr>3.53</nr>
              <absp>1000</absp>
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
                  <block dr="4" dz="0.01593" material="AlAs"/><block dr="6" dz="0.01593" material="AlxOy"/>
                </shelf>
                <block name="x" dr="10" dz="0.00000" material="AlGaAs"/>
                <block dr="10" dz="0.13649" material="GaAs"/>
                <shelf name="QW">
                  <block name="active" dr="4" dz="0.00500" material="InGaAs"/><block dr="6" dz="0.00500" material="InGaAs"/>
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
              <optical name="efm" lib="effective" solver="EffectiveFrequencyCyl">
                <geometry ref="vcsel"/>
                <mode m="0"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.slv.efm
        self.solver.lam0 = 980.

    def testField(self):
        axis0 = linspace(0.001, 39.999, 20000)
        #axis0 = linspace(0.01, 3.99, 200)
        #axis1  = [ self.manager.geo.layers.bbox.lower.z-1e-6, 0.0025, self.manager.geo.layers.bbox.upper.z+-1e-6 ]
        axis1  = [ self.manager.geo.layers.bbox.upper.z+-1e-6 ]
        dr = axis0[1]-axis0[0]
        msh = mesh.Rectilinear2D(axis0, axis1)
        self.solver.compute(980.)
        field = self.solver.outIntensity(msh).array[:,-1]
        integral = 2e-12 * pi * sum(field * msh.axis0) * dr
        power = 1e3 * Z0 * integral # 1e3: W -> mW
        self.assertAlmostEqual(power, 1., 4)


