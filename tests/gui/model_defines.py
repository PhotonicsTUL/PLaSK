import sys, os
import unittest
sys.path.insert(0, os.path.abspath('../..'))

from gui_test_utils import GUITestCase
from gui.model.defines import DefinesModel

from lxml import etree


class TestGUIModelDefines(GUITestCase):

    def setUp(self):
        self.defines = DefinesModel()
        self.defines.entries.append(DefinesModel.Entry("def1", "val1"))
        self.defines.entries.append(DefinesModel.Entry("def2", "val2", ["def 2 comment"]))

    def test_name_to_index(self):
        self.assertEqual(self.defines.name_to_index("def1"), 0)
        self.assertEqual(self.defines.name_to_index("def2"), 1)
        self.assertEqual(self.defines.name_to_index("unexisted"), -1)

    def test_get(self):
        self.assertEqual(len(self.defines.entries), 2)
        self.assertEqual(self.defines.get(0, 0), "def1")
        self.assertEqual(self.defines.get(1, 0), "val1")
        self.assertEqual(self.defines.get(2, 0), [])
        self.assertEqual(self.defines.get(0, 1), "def2")
        self.assertEqual(self.defines.get(1, 1), "val2")
        self.assertEqual(self.defines.get(2, 1), ["def 2 comment"])
        with self.assertRaises(IndexError): self.defines.get(3, 0)
        with self.assertRaises(IndexError): self.defines.get(0, 3)

    def test_set(self):
        self.defines.set(0, 0, "changed def1")
        self.defines.set(1, 1, "changed val2")
        self.assertEqual(self.defines.entries[0].name, "changed def1")
        self.assertEqual(self.defines.entries[0].value, "val1")
        self.assertEqual(self.defines.entries[1].name, "def2")
        self.assertEqual(self.defines.entries[1].value, "changed val2")
        with self.assertRaises(IndexError): self.defines.set(3, 0, "ignored")
        with self.assertRaises(IndexError): self.defines.set(0, 3, "ignored")

    def test_columnCount(self):
        self.assertEqual(self.defines.columnCount(), 2)

    def test_make_xml_element(self):
        self.assertEqualXML(self.defines.make_xml_element(),
            '<defines><define name="def1" value="val1"/><!--def 2 comment--><define name="def2" value="val2"/></defines>')

    def test_load_xml_element(self):
        self.defines.load_xml_element(etree.XML(
            '<defines><!--comment--><define name="x1" value="y1"/><!--last--></defines>'))
        self.assertEqual(len(self.defines.entries), 1)
        self.assertEqual(self.defines.entries[0].name, "x1")
        self.assertEqual(self.defines.entries[0].value, "y1")
        self.assertEqual(self.defines.entries[0].comments, ["comment"])
        self.assertEqual(self.defines.endcomments, ["last"])

    def test_create_info(self):
        self.assertEqual(len(self.defines.create_info()), 0)

    def test_create_info_with_no_name_value(self):
        self.defines.entries[0].name = ''   # no required name
        self.defines.entries[1].value = ''  # no required value
        info = self.defines.create_info()
        self.assertEqual(len(info), 2)
        self.assertCountEqual(info[0].rows + info[1].rows, (0, 1))

    def test_create_info_with_duplicated(self):
        self.defines.entries[0].name = self.defines.entries[1].name   # duplicated name
        info = self.defines.create_info()
        self.assertEqual(len(info), 1)
        self.assertCountEqual(info[0].rows, (0, 1))


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())


