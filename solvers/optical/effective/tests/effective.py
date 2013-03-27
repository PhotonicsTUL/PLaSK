#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.effective import EffectiveIndex2D

@material.simple
class Glass(material.Material):
    def Nr(self, wl, T): return 1.3

@material.simple
class LowConstrast(material.Material):
    def Nr(self, wl, T): return 1.05

class EffectiveIndex2D_Test(unittest.TestCase):

    def setUp(self):
        self.solver = EffectiveIndex2D("test_eim")
        rect = geometry.Rectangle(0.75, 0.5, Glass())
        space = geometry.Cartesian2D(rect, left="mirror")
        self.solver.geometry = space

    def testBasic(self):
        self.assertEqual( self.solver.id, "test_eim:optical.EffectiveIndex2D" )

    def testExceptions(self):
        with self.assertRaisesRegexp(TypeError, r"^No wavelength set nor its provider connected$"):
            self.solver.inWavelength()
        with self.assertRaisesRegexp(ValueError, r"^Effective index cannot be provided now$"):
            self.solver.outNeff()
        with self.assertRaisesRegexp(ValueError, r"^Light intensity cannot be provided now$"):
            self.solver.outIntensity(mesh.Rectilinear2D([1,2],[3,4]))

    def testSymmetry(self):
        self.assertIsNone( self.solver.symmetry )
        self.solver.symmetry = "-"
        self.assertEqual( self.solver.symmetry, "negative" )

    def testReceivers(self):
        self.solver.inWavelength = 850.
        self.assertEqual( self.solver.inWavelength(), 850. )

    def testComputations(self):
        self.solver.inWavelength = 1000.
        self.solver.symmetry = "+"
        self.solver.polarization = "TE"
        # self.assertAlmostEqual( self.solver.compute(1.15), 1.1502818, 6) # effective index
        self.assertAlmostEqual( self.solver.compute(1.15), 1.2035235, 6) # weighted index
        self.solver.polarization = "TM"
        # self.assertAlmostEqual( self.solver.compute(1.10), 1.1156605, 6) # effective index
        self.assertAlmostEqual( self.solver.compute(1.10), 1.1156605, 6) # weighted index

    def testMesh(self):
        mesh = self.solver.mesh


class EffectiveFrequencyCyl_Test(unittest.TestCase):

    def setUp(self):
        self.solver = EffectiveIndex2D("test_efm")
        R = 1.0
        rect = geometry.Rectangle(R, 3.0, LowConstrast())
        self.solver.geometry = geometry.Cartesian2D(rect)
        self.solver.mesh = mesh.Rectilinear2D([0., R], [1.0, 2.0])

    #def testBessel(self):


