#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.slab import FourierReflectionCyl


class LayerSet_Test(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'rz'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask>
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
                <block dr="10" dz="0.13649" material="GaAs" role="opt-cavity"/>
                <shelf name="QW">
                  <block name="active" dr="4" dz="0.00500" material="InGaAs"/><block dr="6" dz="0.00500" material="InGaAs"/>
                </shelf>
                <zero/>
                <block dr="10" dz="0.13649" material="GaAs" role="opt-cavity"/>
                <stack name="bottom-dbr" repeat="29">
                  <block dr="10" dz="0.07955" material="AlGaAs"/>
                  <block dr="10" dz="0.06949" material="GaAs"/>
                </stack>
                <block dr="10" dz="0.07955" material="AlGaAs"/>
                </stack>
              </cylindrical>
            </geometry>
            <solvers>
              <optical name="pwrt" solver="FourierReflectionCyl">
                <geometry ref="vcsel"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.slv.pwrt
        self.mat = self.manager.geo.vcsel.get_material

    def testInterface(self):
        self.solver.set_interface(0.)
        self.assertEqual( self.solver.interface, 60 )
        self.solver.set_interface(self.manager.geo.QW)
        self.assertEqual( self.solver.interface, 61 )

    def testLayers(self):
        self.solver.find_mode(1.)
        layers = [ ' '.join(set([ "%s/%s" % (self.mat(2,z), self.mat(7,z)) for z in l ])) for l in self.solver.layer_sets ]
        stack = list(self.solver.stack)[-1::-1]
        for i in enumerate(layers):
            print("%d: %s" % i)
        print(stack)
        self.assertEqual( layers, ['GaAs/GaAs', 'AlGaAs/AlGaAs', 'GaAs/GaAs', 'InGaAs/InGaAs', 'AlAs/AlxOy', 'air/air'] )
        self.assertEqual( stack, [5] + 25*[0,1] + [4,2,3,2] + 30*[1,0] )

