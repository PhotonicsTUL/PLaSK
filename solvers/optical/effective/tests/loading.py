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

@material.simple()
class LowContrastMaterial(material.Material):
    def Nr(self, wl, T): return 1.3

class EffectiveIndex2D_Test(unittest.TestCase):

    def setUp(self):
        self.manager = Manager()
        self.manager.load("""
        <plask>
            <geometry>
                <cartesian2d name="Space-1" axes="xy">
                    <stack name="Stack-2">
                        <item path="Path-4"><rectangle name="Block-3" dx="5" dy="2" material="GaN" /></item>
                        <again ref="Block-3"/>
                    </stack>
                </cartesian2d>
            </geometry>
            <grids>
                <mesh type="rectangular2d" name="lin">
                    <axis0>1, 2, 3</axis0>
                    <axis1>10 20 30</axis1>
                </mesh>
                <generator type="rectangular2d" method="divide" name="div">
                    <prediv by="4"/>
                    <postdiv by0="2" by1="3"/>
                </generator>
            </grids>
            <solvers>
                <optical lib="effective" solver="EffectiveIndex2D" name="eff1">
                    <geometry ref="Space-1"/>
                    <mesh ref="lin"/>
                    <mode polarization="TM"/>
                    <root tolx="0.1"/>
                </optical>
                <optical solver="EffectiveIndex2D" name="eff2">
                    <geometry ref="Space-1"/>
                    <mesh ref="div"/>
                </optical>
            </solvers>
        </plask>
        """)
        self.solver1 = self.manager.solvers.eff1
        self.solver2 = self.manager.solvers.eff2
        profile = plask.StepProfile(self.manager.geo.Space_1, default=300.)
        profile[self.manager.geo.Block_3] = 320.
        self.solver1.inTemperature = profile.outTemperature

    def testLoadConfigurations(self):
        self.assertEqual(self.solver1.id, "eff1:optical.EffectiveIndex2D")

        self.assertEqual(self.solver1.geometry, self.manager.geo.Space_1)
        self.assertEqual(self.solver2.geometry.item, self.manager.geo.Stack_2)

        self.assertEqual(self.solver1.mesh, self.manager.msh.lin)
        self.assertEqual(self.solver2.mesh, self.manager.msg.div(self.manager.geo.Space_1.item))

        self.assertEqual(self.solver1.polarization, "TM")
        self.assertEqual(self.solver1.root.tolx, 0.1)


    def testProfile(self):
        m = plask.mesh.Rectangular2D(plask.mesh.Regular(1.,), plask.mesh.Regular(1., 5., 2))
        print(list(m))
        print(self.manager.geo.Space_1.get_leafs_bboxes())
        self.assertEqual(list(self.solver1.inTemperature(m)), [320., 300.])
