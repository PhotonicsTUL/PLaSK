#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import slab

plask.config.axes = "xyz"

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
                    <item xcenter="0" ycenter="0"><block dx="10" dy="10" dz="0.01593" material="AlxOy"/></item>
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
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solver.fourier3d
        self.profile = StepProfile(self.solver.geometry)
        self.solver.inGain = self.profile.outGain
        self.solver.refine = 32
        self.solver.size = 7
        self.solver.root.method = 'broyden'
        self.solver.symmetry = 'Ex', 'Ex'

    def testComputations(self):
        m = self.solver.find_mode(lam=979.7)
        self.assertEqual( m, 0 )
        self.assertEqual( len(self.solver.modes), 1 )
        self.assertAlmostEqual( self.solver.modes[m].lam, 979.702, 3 )
        #lams = linspace(978., 981., 101)
        #dets = [abs(self.solver.determinant(lam=lam, dispersive=False)) for lam in lams]
        #plot(lams, dets)
        #show()
