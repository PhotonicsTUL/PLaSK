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
from numpy.testing import assert_allclose

import plask

from plask.util import gradients


plot_exact = True


class TestGradientXpl(unittest.TestCase):

    def setUp(self):
        manager = plask.Manager()
        manager.load('''
        <plask>
            <geometry>
                <cartesian2d name="original2d" axes="x,y">
                <stack>
                    <rectangle material="GaAs" dx="5." dy="1.0"/>
                    <rectangle material-top="Al(0.0)GaAs" material-bottom="Al(1.0)GaAs" dx="5." dy="0.2"/>
                    <rectangle material="AlAs" dx="5." dy="1.0"/>
                    <rectangle material-top="Al(1.0)GaAs" material-bottom="Al(0.0)GaAs" dx="5." dy="0.5"/>
                    <rectangle material="GaAs" dx="5." dy="1.0"/>
                </stack>
                </cartesian2d>
                <cartesian2d name="simplified2d" axes="x,y">
                <copy from="original2d">
                    <simplify-gradients lam="980"/>
                </copy>
                </cartesian2d>
            </geometry>
        </plask>
        ''')
        self.original2d = manager.geo.original2d
        self.simplified2d = manager.geo.simplified2d

    res1 = 3.4029 + 1.5121e-6j
    res2 = 3.0334 + 1.6038e-6j
    d05 = 0.245692
    d02 = 0.101723

    def test1(self):
        mesh = plask.mesh.Rectangular2D([2.], [1.01, 1.24, 1.26, 1.49])
        self.assertAlmostEqual(self.simplified2d.item[1][0][0][0].height, self.d05, 4)
        self.assertAlmostEqual(self.simplified2d.item[1][0][0][0].height + self.simplified2d.item[1][0][1][0].height, 0.5, 5)
        assert_allclose(array(plask.MaterialField(self.simplified2d, mesh).nr(980.)),
                        array([self.res1, self.res1, self.res2, self.res2]),
                        atol=1e-4)

    def test2(self):
        mesh = plask.mesh.Rectangular2D([2.], [2.51, 2.59, 2.61, 2.69])
        self.assertAlmostEqual(self.simplified2d.item[3][0][0][0].height, self.d02, 4)
        self.assertAlmostEqual(self.simplified2d.item[3][0][0][0].height + self.simplified2d.item[3][0][1][0].height, 0.2, 5)
        assert_allclose(array(plask.MaterialField(self.simplified2d, mesh).nr(980.)),
                        array([self.res2, self.res2, self.res1, self.res1]),
                        atol=1e-4)

    if __name__ == '__main__':
        def testPlot(self):
            mesh = plask.mesh.Rectangular2D([2.], plask.mesh.Regular(0.0, 3.7, 3701))
            global plot_exact
            if plot_exact:
                plask.plot_profile(plask.MaterialField(self.original2d, mesh).Nr(980.), color='#1f77b4', label='exact')
                plot_exact = False
            plask.plot_profile(plask.MaterialField(self.simplified2d, mesh).Nr(980.), color='#e5ae38', label='nr')


class TestGradientPython(unittest.TestCase):

    def setUp(self):
        self.original2d = plask.geometry.Stack2D()
        self.original2d.prepend(geometry.Rectangle(5., 1.0, 'GaAs'))
        self.original2d.prepend(geometry.Rectangle(5., 0.2, ('Al(1.0)GaAs', 'Al(0.0)GaAs')))
        self.original2d.prepend(geometry.Rectangle(5., 1.0, 'AlAs'))
        self.original2d.prepend(geometry.Rectangle(5., 0.5, ('Al(0.0)GaAs', 'Al(1.0)GaAs')))
        self.original2d.prepend(geometry.Rectangle(5., 1.0, 'GaAs'))

        self.simplified2d = gradients.simplify_all(self.original2d, 980., linear='eps')

    res1 = 3.4073-3.6809e-7j
    res2 = 3.0385-3.6479e-7j
    d05 = 0.250
    d02 = 0.100

    def test1(self):
        mesh = plask.mesh.Rectangular2D([2.], [1.01, 1.24, 1.26, 1.49])
        self.assertAlmostEqual(self.simplified2d[1][0][0][0].height, self.d05, 4)
        self.assertAlmostEqual(self.simplified2d[1][0][0][0].height + self.simplified2d[1][0][1][0].height, 0.5, 5)
        assert_allclose(array(plask.MaterialField(self.simplified2d, mesh).nr(980.)),
                        array([self.res1, self.res1, self.res2, self.res2]),
                        atol=1e-4)

    def test2(self):
        mesh = plask.mesh.Rectangular2D([2.], [2.51, 2.59, 2.61, 2.69])
        self.assertAlmostEqual(self.simplified2d[3][0][0][0].height, self.d02, 4)
        self.assertAlmostEqual(self.simplified2d[3][0][0][0].height + self.simplified2d[3][0][1][0].height, 0.2, 5)
        assert_allclose(array(plask.MaterialField(self.simplified2d, mesh).nr(980.)),
                        array([self.res2, self.res2, self.res1, self.res1]),
                        atol=1e-4)

    if __name__ == '__main__':
        def testPlot(self):
            mesh = plask.mesh.Rectangular2D([2.], plask.mesh.Regular(0.0, 3.7, 3701))
            global plot_exact
            if plot_exact:
                plask.plot_profile(plask.MaterialField(self.original2d, mesh).Nr(980.), color='#1f77b4', label='exact')
                plot_exact = False
            plask.plot_profile(plask.MaterialField(self.simplified2d, mesh).Nr(980.), color='#d62728', label='exp')


if __name__ == '__main__':
    test = unittest.main(exit=False)
    plask.legend(loc='lower right')
    plask.show()
    sys.exit(not test.result.wasSuccessful())
