#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest


import plask


class Manager(unittest.TestCase):

    def setUp(self):
        self.manager = plask.Manager()
        self.manager.read('''
            <geometry>
                <2d name="Space 1" axes="xy">
                    <stack repeat="2" name="Stack 2">
                        <child path="Path 4"><rectangle name="Block 3" x="4" y="2" material="GaN" /></child>
                        <ref name="Block 3"/>
                    </stack>
                </2d>
            </geometry>
        ''')

    def testBasic(self):
        self.assertEqual( len(self.manager.elements), 2 )
        self.assertEqual( type(self.manager.elements["Block 3"]), plask.geometry.Block2D )
        self.assertEqual( list(self.manager.elements["Stack 2"].getLeafsBBoxes()),
            [plask.geometry.Box2D(0,0,4,2), plask.geometry.Box2D(0,2,4,4), plask.geometry.Box2D(0,4,4,6), plask.geometry.Box2D(0,6,4,8)])
        self.assertEqual( type(self.manager.geometries.Space_1), plask.geometry.Cartesian2D )
        self.assertEqual( len(self.manager.ph), 1 )
        with self.assertRaises(KeyError): self.manager.elements["nonexistent"]

    def testDictionaries(self):
        self.assertEqual( list(self.manager.el), ["Block 3", "Stack 2"] )

    def testExport(self):
        self.manager.export(globals())
        self.assertIn( "Space 1", ge )
        self.assertEqual( type(ge.Space_1), plask.geometry.Cartesian2D )
