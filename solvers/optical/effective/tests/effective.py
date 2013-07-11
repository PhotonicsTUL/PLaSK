#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.effective import EffectiveIndex2D, EffectiveFrequencyCyl

@material.simple
class Glass(material.Material):
    def Nr(self, wl, T): return 1.3

@material.simple
class Cladding(material.Material):
    def Nr(self, wl, T): return 1.28

class EffectiveIndex2D_Test(unittest.TestCase):

    def setUp(self):
        self.solver = EffectiveIndex2D("test_eim")
        rect = geometry.Rectangle(0.75, 0.5, Glass())
        space = geometry.Cartesian2D(rect, left="mirror")
        self.solver.geometry = space

    def testBasic(self):
        self.assertEqual( self.solver.id, "test_eim:optical.EffectiveIndex2D" )

    def testExceptions(self):
        with self.assertRaisesRegexp(TypeError, r"^No provider nor value for wavelength$"):
            self.solver.inWavelength()
        with self.assertRaisesRegexp(ValueError, r"^Effective index cannot be provided now$"):
            self.solver.outNeff()
        with self.assertRaisesRegexp(ValueError, r"^Intensity profile cannot be provided now$"):
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
        self.assertAlmostEqual( self.solver.compute(1.15), 1.147, 3)
        self.solver.polarization = "TM"
        self.assertAlmostEqual( self.solver.compute(1.11), 1.111, 3)

    def testMesh(self):
        mesh = self.solver.mesh


#class EffectiveFrequencyCyl_Test(unittest.TestCase):

    #def setUp(self):
        #self.solver = EffectiveFrequencyCyl("test_efm")
        #R = 1.0
        #shelf = geometry.Shelf2D()
        #shelf.append(geometry.Rectangle(R, 1.0, Glass()))
        #shelf.append(geometry.Rectangle(3*R, 1.0, Cladding()))
        #self.solver.geometry = geometry.Cartesian2D(shelf)
        #self.solver.mesh = mesh.Rectilinear2D([0., R], [0., 1.0])

    #def testVertical(self):
        #self.solver.k0 =
        #self.solver.get_stripe_determinant_v


    ##def testBessel(self):


