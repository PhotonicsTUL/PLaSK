#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *
import matplotlib

from plask import *
from plask import material, geometry, mesh
from optical.slab import Fourier2D


class TempGradientTest(unittest.TestCase):

    def setUp(self):
        plask.config.axes = 'xy'
        self.manager = plask.Manager()
        self.manager.load('''
          <plask>
            <materials>
              <material name="termic" base="semiconductor">
                <nr>T</nr>
              </material>
            </materials>
            <geometry>
              <cartesian2d axes="xy" name="main" left="mirror" right="extend" bottom="GaAs">
                <stack name="layers">
                  <rectangle dx="2" dy="18" material="termic"/>
                  <rectangle dx="2" dy="2" material="termic"/>
                  <zero/>
                </stack>
              </cartesian2d>
            </geometry>
            <grids>
              <mesh name="term" type="rectangular2d">
                <axis0>1</axis0>
                <axis1 start="0" stop="20" num="2001"/>
              </mesh>
            </grids>
            <solvers>
              <optical name="fourier" solver="Fourier2D">
                <geometry ref="main"/>
                <expansion temp-diff="1.0" temp-dist="0.1"/>
              </optical>
            </solvers>
          </plask>''')
        self.solver = self.manager.solvers.fourier
        self.solver.wavelength = 1000.
        self.temp = Data(300. + 4. / linspace(8., 1., len(self.manager.msg.term.axis1)), self.manager.msg.term)
        self.solver.inTemperature = self.temp


    def testLayers(self):
        stack = list(self.solver.stack)
        vaxis = self.solver.layer_edges
        centers = self.solver.layer_centers
        self.assertEqual( stack, [0, 1, 1, 3, 4, 5, 2] )
        self.assertEqual( str(vaxis), '[0, 2, 15.45, 18.35, 19.6, 20]' )
        if __name__ == '__main__':
            temp = self.temp.interpolate(mesh.Rectangular2D([0.5], vaxis), 'linear')
            print(stack)
            print(list(vaxis))
            cmap = matplotlib.cm.get_cmap()
            f = 1. / (len(set(stack))-1)
            axvspan(-2., vaxis[0], color=cmap(f*stack[0]), alpha=0.5)
            for i,c in enumerate(stack[1:-1]):
                axvspan(vaxis[i], vaxis[i+1], color=cmap(f*c), alpha=0.5)
            axvspan(vaxis[-1], 22., color=cmap(f*stack[-1]), alpha=0.5)
            plot(centers, self.temp.interpolate(mesh.Rectangular2D([0.5], centers), 'linear'), 'r.')
            for v in vaxis:
                axvline(v, color='k', lw=0.5)
            for t in temp:
                axhline(t, color='k', lw=0.5, alpha=0.5)
            plot_profile(self.temp, color='r')
            xlim(-1., 21.)

if __name__ == '__main__':
    test = unittest.main(exit=False)
    show()
    sys.exit(not test.result.wasSuccessful())
