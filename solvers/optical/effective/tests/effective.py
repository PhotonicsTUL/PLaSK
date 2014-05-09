#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import effective

@material.simple
class Glass(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.3

@material.simple
class Cladding(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.28

class EffectiveIndex(unittest.TestCase):

    def setUp(self):
        self.solver = effective.EffectiveIndex2D("eim")
        rect = geometry.Rectangle(0.75, 0.5, Glass())
        space = geometry.Cartesian2D(rect, left="mirror")
        self.solver.geometry = space

    def testBasic(self):
        self.assertEqual( self.solver.id, "eim:optical.EffectiveIndex2D" )

    def testExceptions(self):
        with self.assertRaisesRegexp(ValueError, r"^Effective index \[0\] cannot be provided now$"):
            self.solver.outNeff(0)
        with self.assertRaisesRegexp(ValueError, r"^Optical field magnitude \[0\] cannot be provided now$"):
            self.solver.outLightMagnitude(0, mesh.Rectilinear2D([1,2],[3,4]))

    def testComputations(self):
        self.solver.wavelength = 1000.
        self.solver.polarization = "TE"
        self.assertAlmostEqual( self.solver.modes[self.solver.find_mode(1.15, '+')].neff, 1.1465, 3 )
        self.solver.polarization = "TM"
        self.assertAlmostEqual( self.solver.modes[self.solver.find_mode([1.10,1.12], '+')].neff, 1.111, 3)

    def testMesh(self):
        mesh = self.solver.mesh

    def testRefractiveIndex(self):
        self.solver.set_simple_mesh()
        msh = self.solver.mesh.get_midpoints()
        geo = self.solver.geometry
        refr = [geo.get_material(point).Nr(1000., 300.) for point in msh]
        self.assertEqual( [nr[0] for nr in self.solver.outRefractiveIndex(msh, 0.)], refr )


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
            def fun(g):
                self.profile[self.rect2] = g
                m = self.solver.find_mode(1.15)
                return self.solver.modes[m].neff.imag
            gain = brentq(fun, 0., 100.)
            self.assertAlmostEqual(gain, 81.6495, 3)

    def testAbsorptionIntegral(self):
       self.profile[self.rect2] = 81.649513489
       m = self.solver.find_mode(1.15)
       self.solver.modes[m].power = 0.7
       self.assertAlmostEqual( self.solver.get_total_absorption(m), -1.0, 3 )

    def testAbsorbedHeat(self):
        self.profile[self.rect2] = 81.649513489
        m = self.solver.find_mode(1.15)
        self.solver.modes[m].power = 0.7
        box = self.solver.geometry.item.bbox
        msh = mesh.Rectilinear2D(linspace(box.lower.x, box.upper.x, 1000), linspace(box.lower.y, box.upper.y, 1000))
        heat = self.solver.outHeat(msh)
        # 1e-15: µm³->m³ W->mW
        integral = 2e-15 * sum(heat) * (msh.axis0[1] - msh.axis0[0]) * (msh.axis1[1] - msh.axis1[0]) * self.solver.geometry.extrusion.length
        self.assertAlmostEqual( integral, self.solver.get_total_absorption(m), 2 )


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
              <material name="AlxOy" base="semiconductor">
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
                  <block dr="4" dz="0.01593" material="AlAs"/><block dr="6" dz="0.01593" material="AlxOy"/>
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
              <optical name="efm" lib="effective" solver="EffectiveFrequencyCyl">
                <geometry ref="vcsel"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solver.efm
        self.solver.lam0 = 980.
        self.profile = StepProfile(self.solver.geometry)
        self.solver.inGain = self.profile.outGain

    def testField(self):
        axis0 = linspace(0.001, 39.999, 100000)
        #axis0 = linspace(0.01, 3.99, 200)
        #axis1  = [ self.manager.geo.layers.bbox.lower.z-1e-6, 0.0025, self.manager.geo.layers.bbox.upper.z+-1e-6 ]
        axis1  = [ self.manager.geometry.layers.bbox.upper.z+-1e-6 ]
        dr = axis0[1]-axis0[0]
        msh = mesh.Rectilinear2D(axis0, axis1)
        self.solver.find_mode(980., 0)
        self.solver.modes[0].power = 2000.
        field = self.solver.outLightMagnitude(0,msh).array[:,-1]
        integral = 2e-12 * pi * sum(field * msh.axis0) * dr
        self.assertAlmostEqual( integral, 2., 4 )

    def testRefractiveIndex(self):
        self.solver.set_simple_mesh()
        msh = self.solver.mesh.get_midpoints()
        geo = self.solver.geometry
        refr = [geo.get_material(point).Nr(980., 300.) for point in msh]
        self.assertEqual(
            ["%.8g" % nr[0].real for nr in self.solver.outRefractiveIndex(msh, 0.)],
            ["%.8g" % r.real for r in refr]
        )

    def testComputations(self):
        m = self.solver.find_mode(980.)
        self.assertEqual( m, 0 )
        self.assertEqual( len(self.solver.modes), 1 )
        self.assertAlmostEqual( self.solver.modes[m].lam, 979.702, 3 )

    def testThreshold(self):
        try:
            from scipy.optimize import brentq
        except ImportError:
            pass
        else:
            def fun(g):
                self.profile[self.manager.geometry['active']] = g
                m = self.solver.find_mode(980.)
                return self.solver.modes[m].loss
            threshold = brentq(fun, 0., 2000., xtol=1e-6)
            self.assertAlmostEqual( threshold, 1181.7, 1 )

    def testAbsorptionIntegral(self):
        self.profile[self.manager.geometry['active']] = 1181.6834
        m = self.solver.find_mode(980.)
        self.solver.modes[m].power = 2.0
        box = self.solver.geometry.item.bbox
        field = self.solver.outLightMagnitude(m, mesh.Rectilinear2D([0.], [box.lower.z, box.upper.z]))
        total_power = self.solver.modes[m].power * (1. + 3.53 * field[0] / field[1])
        self.assertAlmostEqual( -self.solver.get_total_absorption(m), total_power, 2 )

    def testAbsorbedHeat(self):
        self.profile[self.manager.geometry['active']] = 1181.6834
        m = self.solver.find_mode(980.)
        self.solver.modes[m].power = 2.0
        box = self.solver.geometry.get_object_bboxes(self.manager.geometry['active'])[0]
        msh = mesh.Rectilinear2D(linspace(0., 10., 1000.), [0.5 * (box.lower.z + box.upper.z)])
        heat = self.solver.outHeat(msh).array[:,0]
        # 1e-15: µm³->m³ W->mW
        integral = 2e-15 * pi * sum(heat * msh.axis0) * (box.upper.z - box.lower.z) * (msh.axis0[1] - msh.axis0[0])
        self.assertAlmostEqual( integral, self.solver.get_total_absorption(m), 2 )
