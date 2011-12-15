#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import random

# TODO: write proper test cases

class SimpleDummy(unittest.TestCase):

    def setUp(self):
        '''test case initialization'''
        self.seq = range(10)

    def test_Shuffle(self):
        # make sure the shuffled sequence does not lose any elements
        random.shuffle(self.seq)
        self.seq.sort()
        self.assertEqual(self.seq, range(10))

        # should raise an exception for an immutable sequence
        self.assertRaises(TypeError, random.shuffle, (1,2,3))

    def test_Choice(self):
        element = random.choice(self.seq)
        self.assertTrue(element in self.seq)

    def test_Sample(self):
        with self.assertRaises(ValueError):
            random.sample(self.seq, 20)
        for element in random.sample(self.seq, 5):
            self.assertTrue(element in self.seq)
