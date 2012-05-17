#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import geometry
from plask.optical.effective import EffectiveIndex2D



class EffectiveIndex2D_Test(unittest.TestCase):

    def setUp(self):
        self.module = EffectiveIndex2D()
        rect = geometry.Rectangle(2,1, "GaN")
        stack = geometry.Stack2D()
        for i in range(4): stack.append(rect)
        hint = stack.append(rect)
        for i in range(5): stack.append(rect)
        space = Space2DCartesian(stack, left="mirror")
        self.module.geometry = space

    def testSymmetry(self):
        self.assertIsNone( self.module.symmetry )
        self.module.symmetry = "-"
        self.assertEqual( self.module.symmetry, "negative" )
