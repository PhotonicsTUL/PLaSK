import sys, os
import unittest
sys.path.insert(0, os.path.abspath('../..'))

from gui_test_utils import GUITestCase

from gui.utils.xml import attr_to_xml, xml_to_attr, AttributeReader, OrderedTagReader
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


class TestAttributeReader(GUITestCase):

    @staticmethod
    def _construct_reader():
        return AttributeReader(etree.XML('<element attr1="val1" attr2="val2" attr3="val3"/>'))

    def test_unexpected_attributes(self):
        with self.assertRaisesRegexp(ValueError, 'unexpected.*attr'):
            with TestAttributeReader._construct_reader(): pass

    def test_unexpected_attribute(self):
        with self.assertRaisesRegexp(ValueError, 'unexpected.*attr2'):
            with TestAttributeReader._construct_reader() as r:
                self.assertEqual(r.get('attr1'), 'val1')
                self.assertEqual(r.get('attr3'), 'val3')

    def test_require(self):
        with TestAttributeReader._construct_reader() as r:
            self.assertEqual(r.require('attr1'), 'val1')
            self.assertEqual(r.require('attr2'), 'val2')
            with (self.assertRaisesRegexp(KeyError, '"unexisted_attr" is expected in tag <element>')):
                r.require('unexisted_attr')
            self.assertEqual(r.require('attr3'), 'val3')

    def test_get_and_len_attrib(self):
        with TestAttributeReader._construct_reader() as r:
            self.assertEqual(len(r), 3)
            self.assertEqual(r.get('attr1'), 'val1')
            self.assertEqual(r.get('attr2'), 'val2')
            self.assertEqual(r.get('unexisted_attr'), None)
            self.assertEqual(r.get('unexisted_attr', 'my_default'), 'my_default')
            self.assertEqual(r.get('attr3'), 'val3')
            self.assertEqual(len(r), 3)
            self.assertIs(r.attrib, r)

    def test_contains(self):
        with TestAttributeReader._construct_reader() as r:
            self.assertTrue('attr1' in r)
            self.assertFalse('unexisted_attr' in r)
            self.assertTrue('attr2' in r)
            self.assertTrue('attr3' in r)

    def test_getitem(self):
        with TestAttributeReader._construct_reader() as r:
            self.assertEqual(r['attr1'], 'val1')
            self.assertEqual(r['attr2'], 'val2')
            self.assertEqual(r['attr3'], 'val3')

    def test_mark_read_some(self):
        with self.assertRaisesRegexp(ValueError, 'unexpected.*attr3'):
            with TestAttributeReader._construct_reader() as r:
                r.mark_read('attr1', 'attr2')

    def test_mark_read(self):
        with TestAttributeReader._construct_reader() as r:
            r.mark_read('attr1', 'attr2', 'attr3')

    def test_require_all_read(self):
        with TestAttributeReader._construct_reader() as r:
            r.mark_read('attr1', 'attr2')
            with self.assertRaisesRegexp(ValueError, 'unexpected.*attr3'):
                r.require_all_read()
            self.assertEqual(r.require('attr3'), 'val3')


class TestOrderedTagReader(GUITestCase):

    @staticmethod
    def _construct_reader():
        return OrderedTagReader(etree.XML('<parent><a/><b/><!--comment--><c><child/></c><!--comment--><d/></parent>'))

    def test_unexpected_child(self):
        with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child.*a'):
            with TestOrderedTagReader._construct_reader(): pass

    def test_unexpected_child_after_some_reads(self):
        with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child.*a'):
            with TestOrderedTagReader._construct_reader() as r:
                self.assertEqualXML(r.get(), '<a/>')
                self.assertEqualXML(r.get(), '<b/>')

    def test_get(self):
        with TestOrderedTagReader._construct_reader() as r:
            self.assertEqualXML(r.get(), '<a/>')
            self.assertEqualXML(r.get('b'), '<b/>')
            self.assertIs(r.get('child', 'other', 'b', 'd'), None)
            self.assertEqualXML(r.get('other', 'c'), '<c><child/></c>')
            self.assertIs(r.get('c', 'a', 'b'), None)
            self.assertEqualXML(r.get('c', 'd', 'b'), '<d/>')


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())

