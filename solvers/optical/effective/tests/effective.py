#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.effective import EffectiveIndex2D, EffectiveFrequencyCyl

@material.simple
class Glass(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.3

@material.simple
class Cladding(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.28

class EffectiveIndex2D_Test(unittest.TestCase):

    def setUp(self):
        self.solver = EffectiveIndex2D("eim")
        rect = geometry.Rectangle(0.75, 0.5, Glass())
        space = geometry.Cartesian2D(rect, left="mirror")
        self.solver.geometry = space

    def testBasic(self):
        self.assertEqual( self.solver.id, "eim:optical.EffectiveIndex2D" )

    def testExceptions(self):
        with self.assertRaisesRegexp(ValueError, r"^Effective index \[0\] cannot be provided now$"):
            self.solver.outNeff(0)
        with self.assertRaisesRegexp(ValueError, r"^Light intensity \[0\] cannot be provided now$"):
            self.solver.outLightIntensity(0, mesh.Rectangular2D(mesh.Rectilinear([1,2]),mesh.Rectilinear([3,4])))

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


class EffectiveFrequencyCyl_Test(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'rz'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask loglevel="debug">
            <materials>
            <material name="GaAs" base="semiconductor">
              <nr>3.53</nr>
            </material>
            <material name="AlGaAs" base="semiconductor">
              <nr>3.08</nr>
            </material>
            <material name="AlAs" base="semiconductor">
              <nr>2.95</nr>
            </material>
            <material name="InGaAs" base="semiconductor">
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
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solver.efm
        self.solver.lam0 = 980.

    def testField(self):
        axis0 = plask.mesh.Regular(0.001, 39.999, 20000)
        #axis0 = linspace(0.01, 3.99, 200)
        #axis1  = [ self.manager.geo.layers.bbox.lower.z-1e-6, 0.0025, self.manager.geo.layers.bbox.upper.z+-1e-6 ]
        axis1  = plask.mesh.Rectilinear([ self.manager.geometry.layers.bbox.upper.z+-1e-6 ])
        dr = axis0[1]-axis0[0]
        msh = mesh.Rectangular2D(axis0, axis1)
        self.solver.find_mode(980., 0)
        self.solver.modes[0].power = 2000.
        field = self.solver.outLightIntensity(0,msh).array[:,-1]
        integral = 2e-12 * pi * sum(field * msh.axis0) * dr
        self.assertAlmostEqual(integral, 2., 4)

    def testRefractiveIndex(self):
        self.solver.set_simple_mesh()
        msh = self.solver.mesh.get_midpoints()
        geo = self.solver.geometry
        refr = [geo.get_material(point).Nr(980., 300.) for point in msh]
        self.assertEqual(
            ["%.8g%+.8gj" % (nr[0].real,nr[0].imag) for nr in self.solver.outRefractiveIndex(msh, 0.)],
            ["%.8g%+.8gj" % (r.real,r.imag) for r in refr]
        )
