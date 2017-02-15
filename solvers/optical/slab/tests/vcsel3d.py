#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import slab

plask.config.axes = "prz"

class VCSEL(unittest.TestCase):

    def setUp(self):
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
              <cartesian3d axes="xyz" name="vcsel" left="mirror" right="extend" back="mirror" front="extend" bottom="GaAs">
                <clip left="0" back="0">
                  <stack name="layers" xcenter="0" ycenter="0">
                    <block dx="10" dy="10" dz="0.06949" material="GaAs"/>
                    <stack name="top-dbr" repeat="24">
                      <block dx="10" dy="10" dz="0.07955" material="AlGaAs"/>
                      <block dx="10" dy="10" dz="0.06949" material="GaAs"/>
                    </stack>
                    <block name="x1" dx="10" dy="10" dz="0.06371" material="AlGaAs"/>
                    <align bottom="0" name="oxide-layer">
                      <item xcenter="0" ycenter="0"><block dx="10" dy="10" dz="0.01593" material="AlOx"/></item>
                      <item xcenter="0" ycenter="0"><cylinder radius="4" height="0.01593" material="AlAs"/></item>
                    </align>
                    <block name="x" dx="10" dy="10" dz="0.00000" material="AlGaAs"/>
                    <block dx="10" dy="10" dz="0.13649" material="GaAs"/>
                    <align bottom="0" name="QW">
                      <item xcenter="0" ycenter="0"><block dx="10" dy="10" dz="0.00500" material="InGaAs"/></item>
                      <item xcenter="0" ycenter="0"><cylinder name="active" role="gain" radius="4" height="0.00500" material="InGaAs"/></item>
                    </align>
                    <zero/>
                    <block dx="10" dy="10" dz="0.13649" material="GaAs"/>
                    <stack name="bottom-dbr" repeat="29">
                      <block dx="10" dy="10" dz="0.07955" material="AlGaAs"/>
                      <block dx="10" dy="10" dz="0.06949" material="GaAs"/>
                    </stack>
                    <block dx="10" dy="10" dz="0.07955" material="AlGaAs"/>
                  </stack>
                </clip>
              </cartesian3d>
            </geometry>
            <solvers>
              <optical name="fourier3d" solver="Fourier3D">
                <geometry ref="vcsel"/>
                <expansion lam0="980"/>
                <interface object="QW"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solvers.fourier3d
        self.profile = StepProfile(self.solver.geometry)
        self.solver.inGain = self.profile.outGain
        self.solver.refine = 32
        self.solver.size = 5
        self.solver.root.method = 'muller'
        self.solver.symmetry = 'Er', 'Er'

    def testComputations(self):
        m = self.solver.find_mode(lam=979.75)
        self.assertEqual( m, 0 )
        self.assertEqual( len(self.solver.modes), 1 )
        print(self.solver.pmls)
        self.assertAlmostEqual( self.solver.modes[m].lam, 979.678-0.021j, 3 )
        pass

if __name__ == "__main__":
    vcsel = VCSEL('testComputations')
    vcsel.setUp()

    vcsel.solver.initialize()
    z0 = 0.

    box = vcsel.solver.geometry.bbox
    msh = mesh.Rectangular3D([0.], mesh.Regular(0., 7., 201), mesh.Regular(box.lower.z, box.upper.z, 1001))
    mshr = mesh.Rectangular3D([0.], mesh.Regular(0., 7., 201), [z0])

    #lams = linspace(977., 981., 201)
    #dets = vcsel.solver.get_determinant(lam=lams)
    #plot(lams, abs(dets))
    #yscale('log')

    modes = vcsel.solver.find_mode(lam=979.75), vcsel.solver.find_mode(lam=978.48),

    for m in modes:
        print(vcsel.solver.modes[m])

        figure()
        plot_field(vcsel.solver.outLightMagnitude(m, msh, 'fourier'), plane='rz')

        figure()
        plot_profile(vcsel.solver.outLightMagnitude(m, mshr, 'fourier') / vcsel.solver.modes[m].power)
        axvline(box.upper.r, ls=':', color='k')
        axvline(box.upper[1] + vcsel.solver.pmls[1].dist, ls=':', color='k')
        axvline(box.upper[1] + vcsel.solver.pmls[1].dist + vcsel.solver.pmls[1].size, ls=':', color='k')
        xlim(mshr.axis1[0], mshr.axis1[-1])

    show()
