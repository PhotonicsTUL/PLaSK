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
from gui.model.connects import ConnectsModel

from lxml import etree


class TestGUIConnectDefines(GUITestCase):

    def setUp(self):
        self.connects = ConnectsModel()
        self.connects.entries.append(ConnectsModel.Entry("out1", "in1"))
        self.connects.entries.append(ConnectsModel.Entry("out2", "in2", ["connect 2 comment"]))

    def test_get(self):
        self.assertEqual(len(self.connects.entries), 2)
        self.assertEqual(self.connects.get(0, 0), "in1")
        self.assertEqual(self.connects.get(1, 0), "out1")
        self.assertEqual(self.connects.get(2, 0), [])
        self.assertEqual(self.connects.get(0, 1), "in2")
        self.assertEqual(self.connects.get(1, 1), "out2")
        self.assertEqual(self.connects.get(2, 1), ["connect 2 comment"])
        with self.assertRaises(IndexError): self.connects.get(3, 0)
        with self.assertRaises(IndexError): self.connects.get(0, 3)

    def test_set(self):
        self.connects.set(1, 0, "changed out1")
        self.connects.set(0, 1, "changed in2")
        self.assertEqual(self.connects.entries[0].output, "changed out1")
        self.assertEqual(self.connects.entries[0].input, "in1")
        self.assertEqual(self.connects.entries[1].output, "out2")
        self.assertEqual(self.connects.entries[1].input, "changed in2")
        with self.assertRaises(IndexError): self.connects.set(3, 0, "ignored")
        with self.assertRaises(IndexError): self.connects.set(0, 3, "ignored")

    def test_columnCount(self):
        self.assertEqual(self.connects.columnCount(), 2)

    def test_make_xml_element(self):
        self.assertEqualXML(self.connects.make_xml_element(),
            '<connects><connect in="in1" out="out1"/><!--connect 2 comment--><connect in="in2" out="out2"/></connects>')

    def test_load_xml_element(self):
        xml = etree.XML('<connects><!--comment--><connect in="x1" out="y1"/><!--last--></connects>')
        self.connects.load_xml_element(xml)
        self.assertEqual(len(self.connects.entries), 1)
        self.assertEqual(self.connects.entries[0].input, "x1")
        self.assertEqual(self.connects.entries[0].output, "y1")
        self.assertEqual(self.connects.entries[0].comments, ["comment"])
        self.assertEqual(self.connects.endcomments, ["last"])

    def test_create_info(self):
        self.assertEqual(len(self.connects.create_info()), 0)

    def test_create_info_with_no_inout_value(self):
        self.connects.entries[0].input = ''   # no required input
        self.connects.entries[1].output = ''  # no required output
        info = self.connects.create_info()
        self.assertEqual(len(info), 2)
        self.assertCountEqual(info[0].rows + info[1].rows, (0, 1))



if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())


