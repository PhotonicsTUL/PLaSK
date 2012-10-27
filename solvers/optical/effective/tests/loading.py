#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

@material.simple
class LowContrastMaterial(material.Material):
    def nR(self, wl, T): return 1.3

class EffectiveIndex2D_Test(unittest.TestCase):

    def setUp(self):
        self.manager = Manager()
        self.manager.read("""
        <plask>
            <geometry>
                <cartesian2d name="Space-1" axes="xy">
                    <stack name="Stack-2">
                        <child path="Path-4"><rectangle name="Block-3" x="5" y="2" material="GaN" /></child>
                        <ref name="Block-3"/>
                    </stack>
                </cartesian2d>
            </geometry>
            <grids>
                <mesh type="rectilinear2d" name="lin">
                    <axis0>1, 2, 3</axis0>
                    <axis1>10 20 30</axis1>
                </mesh>
                <generator type="rectilinear2d" method="divide" name="div">
                    <prediv by="4"/>
                    <postdiv hor_by="2" vert_by="3"/>
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
            <connects>
              <profile in="eff1.inTemperature">
                <step object="Block-3" value="320."/>
              </profile>
            </connects>
        </plask>
        """)
        self.solver1 = self.manager.slv.eff1
        self.solver2 = self.manager.slv.eff2


    def testLoadConfigurations(self):
        self.assertEqual( self.solver1.id, "eff1:optical.EffectiveIndex2D" )

        self.assertEqual( self.solver1.geometry, self.manager.geo.Space_1 )
        self.assertEqual( self.solver2.geometry.child, self.manager.obj.Stack_2 )

        self.assertEqual( self.solver1.mesh, self.manager.msh.lin )
        self.assertEqual( self.solver2.mesh, self.manager.msg.div(self.manager.geo.Space_1.child) )

        self.assertEqual( self.solver1.polarization, "TM" )
        self.assertEqual( self.solver1.root.tolx, 0.1 )


    def testProfile(self):
        m = plask.mesh.Regular2D((1.,), (1., 5., 2))
        print list(m)
        print self.manager.geo.Space_1.getLeafsBBoxes()
        self.assertEqual( list(self.solver1.inTemperature(m)), [320., 300.] )