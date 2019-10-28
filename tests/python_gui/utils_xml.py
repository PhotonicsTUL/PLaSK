import sys, os
import unittest
sys.path.insert(0, os.path.abspath('../..'))

from gui_test_utils import GUITestCase

from gui.utils.xml import attr_to_xml, xml_to_attr
from lxml import etree

class TestGUIUtilsXML(GUITestCase):

    class _SomeClass: pass

    def setUp(self):
        self.element = etree.Element('el')
        self.obj = TestGUIUtilsXML._SomeClass()

    def test_attr_to_xml(self):
        self.obj.attr1 = 'a1v'
        self.obj.attr2 = 'a2v'
        self.obj.attr_to_ignore = None
        attr_to_xml(self.obj, self.element,
                    'attr1', 'attr2', 'attr3', 'attr_to_ignore',
                    attr2='default value to ignore',
                    attr3='default value to set')
        self.assertEqualXML(self.element,
                            '<el attr1="a1v" attr2="a2v" attr3="default value to set"/>')

    def test_xml_to_attr(self):
        self.element.attrib['attr1'] = 'attr1_value'
        self.element.attrib['attr2'] = 'attr2_value'
        self.obj.attr2 = 'to override'
        xml_to_attr(self.element, self.obj, 'attr1', 'attr2', 'other_attr')
        self.assertEqual(self.obj.attr1, 'attr1_value')
        self.assertEqual(self.obj.attr2, 'attr2_value')
        self.assertEqual(self.obj.other_attr, None)

    def test_xml_to_attr_extra_attr_error(self):
        self.element.attrib['attr1'] = 'attr1_value'
        self.element.attrib['unread'] = 'to cause error'
        with self.assertRaises(ValueError): xml_to_attr(self.element, self.obj, 'attr1')


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())

