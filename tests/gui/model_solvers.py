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
from gui.model.solvers import FilterSolver
from gui.model.solvers.bconds import RectangularBC

from lxml import etree

class TestGUIModelSolversFilter(GUITestCase):

    def setUp(self):
        self.filter = FilterSolver('Temperature', 'filter_name')
        self.filter.geometry = 'geometry_name'

    def test_make_xml_element(self):
        self.assertEqualXML(self.filter.make_xml_element(),
            '<filter for="Temperature" name="filter_name" geometry="geometry_name"/>')

    def test_load_xml_element(self):
        self.filter.load_xml_element(etree.XML(
            '''<filter for="Luminescence" name="name_from_XML" geometry="geometry_from_XML">
                <!-- to ignore --></filter>'''
        ))
        self.assertEqual(self.filter.what, 'Luminescence')
        self.assertEqual(self.filter.name, 'name_from_XML')
        self.assertEqual(self.filter.geometry, 'geometry_from_XML')



class TestGUIModelSolversRectangularBCPlaceSide(GUITestCase):

    def setUp(self):
        self.place = RectangularBC.PlaceSide('right', 'obj_name', 'path_name')

    def test_make_xml_element(self):
        self.assertEqualXML(self.place.make_xml_element(),
                            '<place side="right" object="obj_name" path="path_name"/>')

    def test_eq(self):
        self.assertEqual(self.place, RectangularBC.PlaceSide('right', 'obj_name', 'path_name'))
        self.assertNotEqual(self.place, RectangularBC.PlaceSide('left', 'obj_name', 'path_name'))
        self.assertNotEqual(self.place, RectangularBC.PlaceSide('right', 'different_obj_name', 'path_name'))
        self.assertNotEqual(self.place, RectangularBC.PlaceSide('right', 'obj_name', 'different_path_name'))
        self.assertNotEqual(self.place, 'different type')

    def test_copy_from(self):
        second_place = RectangularBC.PlaceSide('left')
        second_place.copy_from(self.place)
        self.assertEqual(second_place, RectangularBC.PlaceSide('left', 'obj_name', 'path_name'))

    def test_label(self):
        self.assertEqual(self.place.label, 'Right')


class TestGUIModelSolversRectangularBCPlaceLine(GUITestCase):

    def setUp(self):
        self.place = RectangularBC.PlaceLine('horizontal', '1', '2', '10')

    def test_make_xml_element(self):
        self.assertEqualXML('<place line="horizontal" at="1" start="2" stop="10"/>',
                            self.place.make_xml_element())

    def test_eq(self):
        self.assertEqual(self.place, RectangularBC.PlaceLine('horizontal', '1', '2', '10'))
        self.assertNotEqual(self.place, RectangularBC.PlaceLine('vertical', '1', '2', '10'))
        self.assertNotEqual(self.place, RectangularBC.PlaceLine('horizontal', '5', '2', '10'))
        self.assertNotEqual(self.place, RectangularBC.PlaceLine('horizontal', '1', '5', '10'))
        self.assertNotEqual(self.place, RectangularBC.PlaceLine('horizontal', '1', '2', '5'))
        self.assertNotEqual(self.place, 'different type')

    def test_copy_from(self):
        second_place = RectangularBC.PlaceLine('vertical')
        second_place.copy_from(self.place)
        self.assertEqual(second_place, RectangularBC.PlaceLine('vertical', '1', '2', '10'))

    def test_label(self):
        self.assertEqual(self.place.label, 'Horizontal Line')


class TestGUIModelSolversRectangularBC(GUITestCase):

    def setUp(self):
        self.bc = RectangularBC()


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
