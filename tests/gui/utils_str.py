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

import sys, os
import unittest
sys.path.insert(0, os.path.abspath('../..'))

from gui_test_utils import GUITestCase
from gui.utils.str import empty_to_none, none_to_empty, html_to_tex

class TestGUIUtilsXML(GUITestCase):

    def test_empty_to_none(self):
        self.assertEqual(empty_to_none(''), None)
        self.assertEqual(empty_to_none(' \t\n '), None)
        self.assertEqual(empty_to_none('abc'), 'abc')

    def test_none_to_empty(self):
        self.assertEqual(none_to_empty(None), '')
        self.assertEqual(none_to_empty('abc'), 'abc')

    def test_html_to_tex(self):
        self.assertEqual(html_to_tex("aa<br/>bb"), "aa\nbb")
        self.assertEqual(html_to_tex("aa<i>bb</i>cc"), "aa$bb$cc")
        self.assertEqual(html_to_tex("aa<i></i>bb"), "aabb")
        self.assertEqual(html_to_tex("aa<sub>bb</sub>cc"), "aa$_{bb}$cc")
        self.assertEqual(html_to_tex("aa<sup>bb</sup>cc"), "aa$^{bb}$cc")
        self.assertEqual(html_to_tex("aa<sup>bb</sup><sub>cc</sub>dd"), "aa$^{bb}_{cc}$dd")
        self.assertEqual(html_to_tex("aa<i>bb</i>cc<sub>dd</sub><sup>ee</sup><br/><i></i>f"), "aa$bb$cc$_{dd}^{ee}$\nf")


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
