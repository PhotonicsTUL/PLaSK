<plask loglevel="detail">

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
  <cartesian3d name="vcsel" axes="xyz" back="mirror" front="extend" left="mirror" right="extend" bottom="GaAs">
    <clip back="0" left="0">
      <stack name="layers" xcenter="0" ycenter="0">
        <cuboid material="GaAs" dx="10" dy="10" dz="0.06949"/>
        <stack name="top-dbr" repeat="24">
          <cuboid material="AlGaAs" dx="10" dy="10" dz="0.07955"/>
          <cuboid material="GaAs" dx="10" dy="10" dz="0.06949"/>
        </stack>
        <cuboid name="x1" material="AlGaAs" dx="10" dy="10" dz="0.06371"/>
        <align name="oxide-layer" bottom="0">
          <item xcenter="0" ycenter="0">
            <cuboid material="AlOx" dx="10" dy="10" dz="0.01593"/>
          </item>
          <item xcenter="0" ycenter="0">
            <cylinder material="AlAs" radius="4" height="0.01593"/>
          </item>
        </align>
        <cuboid name="x" material="AlGaAs" dx="10" dy="10" dz="0.00000"/>
        <cuboid material="GaAs" dx="10" dy="10" dz="0.13649"/>
        <align name="QW" bottom="0">
          <item xcenter="0" ycenter="0">
            <cuboid material="InGaAs" dx="10" dy="10" dz="0.00500"/>
          </item>
          <item xcenter="0" ycenter="0">
            <cylinder name="active" role="gain" material="InGaAs" radius="4" height="0.00500"/>
          </item>
        </align>
        <zero/>
        <cuboid material="GaAs" dx="10" dy="10" dz="0.13649"/>
        <stack name="bottom-dbr" repeat="29">
          <cuboid material="AlGaAs" dx="10" dy="10" dz="0.07955"/>
          <cuboid material="GaAs" dx="10" dy="10" dz="0.06949"/>
        </stack>
        <cuboid material="AlGaAs" dx="10" dy="10" dz="0.07955"/>
      </stack>
    </clip>
  </cartesian3d>
</geometry>

<solvers>
  <optical name="fourier3d" solver="Fourier3D" lib="slab">
    <geometry ref="vcsel"/>
    <expansion lam0="980"/>
    <interface object="QW"/>
  </optical>
</solvers>

<script><![CDATA[
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import slab

plask.config.axes = "xyz"


class VCSEL:

    def setUp(self):
        self.profile = StepProfile(fourier3d.geometry)
        fourier3d.inGain = self.profile.outGain
        fourier3d.refine = 32
        fourier3d.size = 5
        fourier3d.root.method = 'broyden'
        fourier3d.symmetry = 'Ex', 'Ex'
        fourier3d.rule = 'new'


class NewRule(VCSEL, unittest.TestCase):

    def testComputations(self):
        m = fourier3d.find_mode(lam=979.75)
        self.assertEqual(m, 0)
        self.assertEqual(len(fourier3d.modes), 1)
        self.assertAlmostEqual(fourier3d.modes[m].lam, 979.686-0.0210j, 3)


class Old1Rule(VCSEL, unittest.TestCase):

    def testComputations(self):
        fourier3d.rule = 'old1'
        m = fourier3d.find_mode(lam=979.75)
        self.assertEqual(m, 0)
        self.assertEqual(len(fourier3d.modes), 1)
        self.assertAlmostEqual(fourier3d.modes[m].lam, 979.678-0.0210j, 3)


class Old2Rule(VCSEL, unittest.TestCase):

    def testComputations(self):
        fourier3d.rule = 'old2'
        m = fourier3d.find_mode(lam=979.75)
        self.assertEqual(m, 0)
        self.assertEqual(len(fourier3d.modes), 1)
        self.assertAlmostEqual(fourier3d.modes[m].lam, 979.678-0.0227j, 3)


if __name__ == "__main__":
    vcsel = NewRule('testComputations')
    vcsel.setUp()

    fourier3d.initialize()
    z0 = 0.

    box = fourier3d.geometry.bbox
    msh = mesh.Rectangular3D([0.], mesh.Regular(0., 7., 201), mesh.Regular(box.lower.z, box.upper.z, 1001))
    mshr = mesh.Rectangular3D([0.], mesh.Regular(0., 7., 201), [z0])

    #lams = linspace(977., 981., 201)
    #dets = fourier3d.get_determinant(lam=lams)
    #plot(lams, abs(dets))
    #yscale('log')

    modes = fourier3d.find_mode(lam=979.75), fourier3d.find_mode(lam=978.48),

    for m in modes:
        print_log('result', fourier3d.modes[m])

        figure()
        plot_field(fourier3d.outLightMagnitude(m, msh, 'fourier'), plane='yz')

        figure()
        plot_profile(fourier3d.outLightMagnitude(m, mshr, 'fourier') / fourier3d.modes[m].power)
        axvline(box.upper.y, ls=':', color='k')
        axvline(box.upper[1] + fourier3d.pmls[1].dist, ls=':', color='k')
        axvline(box.upper[1] + fourier3d.pmls[1].dist + fourier3d.pmls[1].size, ls=':', color='k')
        xlim(mshr.axis1[0], mshr.axis1[-1])

    show()
]]></script>

</plask>
