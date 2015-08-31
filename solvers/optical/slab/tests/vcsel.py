#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import slab


plask.config.axes = "rz"


class VCSEL(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'rz'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask loglevel="debug">
            <defines>
              <define name="R" value="20."/>
              <define name="aprt" value="4."/>
            </defines>
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
                <block dr="{R}" dz="0.06949" material="GaAs"/>
                <stack name="top-dbr" repeat="24">
                  <block dr="{R}" dz="0.07955" material="AlGaAs"/>
                  <block dr="{R}" dz="0.06949" material="GaAs"/>
                </stack>
                <block name="x1" dr="{R}" dz="0.06371" material="AlGaAs"/>
                <shelf name="oxide-layer">
                  <block dr="{aprt}" dz="0.01593" material="AlAs"/><block dr="{R-aprt}" dz="0.01593" material="AlOx"/>
                </shelf>
                <block name="x" dr="{R}" dz="0.00000" material="AlGaAs"/>
                <block dr="{R}" dz="0.13649" material="GaAs"/>
                <shelf name="QW">
                  <block name="active" role="gain" dr="{aprt}" dz="0.00500" material="InGaAs"/><block dr="{R-aprt}" dz="0.00500" material="InGaAs"/>
                </shelf>
                <zero/>
                <block dr="{R}" dz="0.13649" material="GaAs"/>
                <stack name="bottom-dbr" repeat="29">
                  <block dr="{R}" dz="0.07955" material="AlGaAs"/>
                  <block dr="{R}" dz="0.06949" material="GaAs"/>
                </stack>
                <block dr="{R}" dz="0.07955" material="AlGaAs"/>
                </stack>
              </cylindrical>
            </geometry>
            <solvers>
              <optical name="bessel" lib="slab" solver="BesselCyl">
                <geometry ref="vcsel"/>
                <interface object="QW"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solver.bessel
        self.profile = StepProfile(self.solver.geometry)
        self.solver.inGain = self.profile.outGain
        self.solver.size = 12

    def testComputations(self):
        m = self.solver.find_mode(980.1)
        self.assertEqual( m, 0 )
        self.assertEqual( len(self.solver.modes), 1 )
        #self.assertAlmostEqual( self.solver.modes[m].lam, 979.702-0.021j, 3 )
        
    def plot_determinant(self):
        lams = linspace(979., 981., 201)
        dets = self.solver.get_determinant(lam=lams, m=1, dispersive=False)
        plot(lams, abs(dets))
        yscale('log')
        
    def plot_field(self):
        self.solver.find_mode(980.1)
        print self.solver.modes[0]
        box = self.solver.geometry.bbox
        msh = mesh.Rectangular2D(linspace(-box.right, box.right, 101),
                                 linspace(box.bottom, box.top, 1001))
        field = self.solver.outElectricField(msh)
        mag = max(abs(field.array.ravel()))
        scale = linspace(-mag, mag, 255)
        figure()
        plot_geometry(self.solver.geometry, mirror=True, color='k', alpha=0.25)
        plot_field(field, scale, comp='r', cmap='bwr')
        gcf().canvas.set_window_title("Er")
        colorbar(use_gridspec=True)
        tight_layout(0.1)
        figure()
        plot_geometry(self.solver.geometry, mirror=True, color='k', alpha=0.25)
        plot_field(field, scale, comp='p', cmap='bwr')
        gcf().canvas.set_window_title("Ep")
        colorbar(use_gridspec=True)
        tight_layout(0.1)
        figure()
        plot_geometry(self.solver.geometry, mirror=True, color='k', alpha=0.25)
        plot_field(field, scale, comp='z', cmap='bwr')
        gcf().canvas.set_window_title("Ez")
        colorbar(use_gridspec=True)
        tight_layout(0.1)


if __name__ == "__main__":
    vcsel = VCSEL('plot_field')
    vcsel.setUp()

    #vcsel.plot_determinant()
    vcsel.plot_field()
    show()
    