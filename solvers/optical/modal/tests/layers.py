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
from numpy.testing import assert_array_almost_equal

from plask import *
from plask import material, geometry, mesh
from optical.modal import Fourier2D


class LayerSetTest(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'rz'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask>
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
              <cartesian2d axes="rz" name="vcsel" left="mirror" right="extend" bottom="GaAs">
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
                  <block dr="10" dz="0.13649" material="GaAs" role="opt-cavity,interface"/>
                  <shelf name="QW">
                    <block name="active" dr="4" dz="0.00500" material="InGaAs"/><block dr="6" dz="0.00500" material="InGaAs"/>
                  </shelf>
                  <block dr="10" dz="0.13649" material="GaAs" role="opt-cavity"/>
                  <zero/>
                  <stack name="bottom-dbr" repeat="29">
                    <block dr="10" dz="0.07955" material="AlGaAs"/>
                    <block dr="10" dz="0.06949" material="GaAs"/>
                  </stack>
                  <block dr="10" dz="0.07955" material="AlGaAs"/>
                </stack>
              </cartesian2d>
            </geometry>
            <solvers>
              <optical name="pwrt" solver="Fourier2D">
                <geometry ref="vcsel"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solvers.pwrt
        self.solver.wavelength = 1000.
        self.mat = self.manager.geo.vcsel.get_material

    def testInterface(self):
        self.assertEqual(self.solver.interface, 62)
        self.solver.set_interface(0.)
        self.assertEqual(self.solver.interface, 60)
        self.solver.set_interface(self.manager.geo.QW)
        self.assertEqual(self.solver.interface, 61)

    def testLayers(self):
        self.solver.initialize()
        #layers = [ ' '.join(set([ "%s/%s" % (self.mat(2,z), self.mat(7,z)) for z in l ])) for l in self.solver.layer_sets ]
        stack = list(self.solver.stack)[-1::-1]
        #for i in enumerate(layers):
            #print("%d: %s" % i)
        #self.assertEqual(layers, ['GaAs/GaAs', 'AlGaAs/AlGaAs', 'GaAs/GaAs', 'InGaAs/InGaAs', 'AlAs/AlOx', 'air/air'])
        print(stack)
        self.assertEqual(stack, [5] + 25*[0,1] + [4,2,3,2] + 30*[1,0])


class MergeTest(unittest.TestCase):

    def setUp(self):
      rect = geometry.Rectangle(1., 1., 'GaAs')
      stack = geometry.Stack2D()
      stack.prepend(rect)
      stack.prepend(rect)
      path = stack.prepend(rect)
      stack.prepend(rect)
      stack.prepend(rect)
      self.solver = Fourier2D('solver')
      self.solver.geometry = geometry.Cartesian2D(stack)
      self.solver.set_interface(rect, path)

    def testMerge(self):
      self.solver.initialize()
      stack = list(self.solver.stack)
      print(stack)
      self.assertEqual(stack, [0, 1, 1, 0])


class TestAnisotropic(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'xyz'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask>
            <materials>
              <material name="Anisotropic" base="semiconductor">
                <Eps>1, 4, 9</Eps>
              </material>
            </materials>
            <geometry>
              <cartesian2d axes="yz" name="test" left="mirror" right="periodic">
                <block dy="10" dz="10" material="Anisotropic"/>
              </cartesian2d>
            </geometry>
            <solvers>
              <optical name="solver" solver="Fourier2D">
                <geometry ref="test"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solvers.solver
        self.solver.wavelength = 1000.
        self.mesh = mesh.Rectangular2D([5.0], [5.0])

    def testEpsilon(self):
      assert_array_almost_equal(diag([1.,4.,9.]), self.solver.outEpsilon(self.mesh)[0])

    def testRefractiveIndex(self):
      self.assertAlmostEqual(3., self.solver.outRefractiveIndex(self.mesh)[0])
      self.assertAlmostEqual(1., self.solver.outRefractiveIndex('ll', self.mesh)[0])
      self.assertAlmostEqual(2., self.solver.outRefractiveIndex('tt', self.mesh, 'default')[0])
      self.assertAlmostEqual(3., self.solver.outRefractiveIndex('vv', self.mesh)[0])
      self.assertAlmostEqual(1., self.solver.outRefractiveIndex('xx', self.mesh)[0])
      self.assertAlmostEqual(2., self.solver.outRefractiveIndex('yy', self.mesh)[0])
      self.assertAlmostEqual(3., self.solver.outRefractiveIndex('zz', self.mesh)[0])


class ProviderTest(unittest.TestCase):

    @staticmethod
    def value(msh, lam, interp):
        return np.array(len(msh) * [lam.imag], dtype=complex)

    def setUp(self):
        plask.config.axes = 'xyz'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask>
            <geometry>
              <cartesian2d axes="yz" name="test2" left="mirror" right="periodic">
                <stack>
                  <block dy="2" dz="2" material="air"/>
                  <block dy="2" dz="2" material="air" role="inEpsilon"/>
                  <block dy="2" dz="2" material="air"/>
                </stack>
              </cartesian2d>
              <cartesian3d axes="xyz" name="test3" back="mirror" front="periodic" left="mirror" right="periodic">
                <stack>
                  <block dx="2" dy="2" dz="2" material="air"/>
                  <block dx="2" dy="2" dz="2" material="air" role="inEpsilon"/>
                  <block dx="2" dy="2" dz="2" material="air"/>
                </stack>
              </cartesian3d>
            </geometry>
            <solvers>
              <optical name="solver2" solver="Fourier2D">
                <geometry ref="test2"/>
              </optical>
              <optical name="solver3" solver="Fourier3D">
                <geometry ref="test3"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver2 = self.manager.solvers.solver2
        self.solver2.wavelength = 1000 + 4j
        self.mesh2 = mesh.Rectangular2D([1.0], [3.0])
        self.solver3 = self.manager.solvers.solver3
        self.solver3.wavelength = 1000 + 9j
        self.mesh3 = mesh.Rectangular3D([1.0], [1.0], [3.0])
        self.solver2.inEpsilon = flow.EpsilonProvider2D(self.value)
        self.solver3.inEpsilon = flow.EpsilonProvider3D(self.value)

    def test2D(self):
       self.assertAlmostEqual(2., self.solver2.outRefractiveIndex(self.mesh2)[0])

    def test3D(self):
       self.assertAlmostEqual(3., self.solver3.outRefractiveIndex(self.mesh3)[0])
