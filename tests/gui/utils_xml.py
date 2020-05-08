import sys, os
import unittest
sys.path.insert(0, os.path.abspath('../..'))

from gui_test_utils import GUITestCase

from gui.utils.xml import attr_to_xml, xml_to_attr, AttributeReader, OrderedTagReader, UnorderedTagReader
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
        return OrderedTagReader(etree.XML(
            '<parent>'
                '<aaa/>'
                '<bbb/>'
                '<!--comment c-->'
                '<ccc>'
                    '<child/>'
                '</ccc>'
                '<!--comment d1-->'
                '<!--comment d2-->'
                '<ddd/>'
                '<!--comment e-->'
            '</parent>'))

    def test_unexpected_child(self):
        with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child.*aaa'):
            with TestOrderedTagReader._construct_reader(): pass

    def test_unexpected_child_after_some_reads(self):
        with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child'):
            with TestOrderedTagReader._construct_reader() as r:
                self.assertEqualXML(r.get(), '<aaa/>')
                self.assertEqualXML(r.get(), '<bbb/>')

    def test_get_and_recent_was_unexpected(self):
        with TestOrderedTagReader._construct_reader() as r:
            self.assertEqualXML(r.get(), '<aaa/>')
            with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child.*aaa'):
                r.recent_was_unexpected()
            self.assertEqualXML(r.get(), '<aaa/>')
            self.assertEqualXML(r.get('bbb'), '<bbb/>')
            self.assertIs(r.get('child', 'other', 'bbb', 'ddd'), None)
            self.assertEqualXML(r.get('other', 'ccc'), '<ccc><child/></ccc>')
            self.assertIs(r.get('ccc', 'aaa', 'bbb'), None)
            self.assertEqualXML(r.get('ccc', 'ddd', 'bbb'), '<ddd/>')
            with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child.*ddd'):
                r.recent_was_unexpected()
            self.assertEqualXML(r.get(), '<ddd/>')
            r.get_comments()

    def test_read_comments(self):
        with TestOrderedTagReader._construct_reader() as r:
            self.assertEqualXML(r.get(), '<aaa/>')
            self.assertEqualXML(r.get(), '<bbb/>')
            c = r.get('ccc')
            self.assertEqual(c.comments, ['comment c'])
            d = r.get('ddd')
            self.assertEqual(d.comments, ['comment d1', 'comment d2'])
            self.assertEqual(list(r._iter_comments()), ['comment e'])


class TestUnorderedTagReader(GUITestCase):

    @staticmethod
    def _construct_reader():
        return UnorderedTagReader(etree.XML(
            '<parent>'
                '<aaa/>'
                '<bbb/>'
                '<!--comment c-->'
                '<ccc>'
                    '<child/>'
                '</ccc>'
                '<!--comment d1-->'
                '<!--comment d2-->'
                '<ddd/>'
                '<!--comment e-->'
            '</parent>'))

    def test_unexpected_child(self):
        with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child'):
            with TestUnorderedTagReader._construct_reader(): pass

    def test_unexpected_child_after_some_reads(self):
        with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child'):
            with TestUnorderedTagReader._construct_reader() as r:
                self.assertEqualXML(r.get('ccc'), '<ccc><child/></ccc>')
                self.assertEqualXML(r.get('aaa'), '<aaa/>')
                self.assertEqualXML(r.get('ccc'), '<ccc><child/></ccc>')    # again

    def test_get_and_mark_read_and_require_all_read(self):
        with TestUnorderedTagReader._construct_reader() as r:
            self.assertEqualXML(r.get('ccc'), '<ccc><child/></ccc>')
            r.mark_read('ddd')
            self.assertEqualXML(r.get('aaa'), '<aaa/>')
            with self.assertRaisesRegexp(ValueError, 'parent.*has unexpected child'):
                r.require_all_read()
            r.mark_read('bbb')
            self.assertEqualXML(r.get('ccc'), '<ccc><child/></ccc>')  # again
            self.assertEqualXML(r.get('bbb'), '<bbb/>')  # again
            r.require_all_read()    # should not throw

    def test_require_and_mark_read(self):
        with TestUnorderedTagReader._construct_reader() as r:
            self.assertEqualXML(r.require('ccc'), '<ccc><child/></ccc>')
            with self.assertRaisesRegexp(KeyError, 'parent.*does not have required.*zzz'):
                r.require('zzz')
            r.mark_read('ddd')
            self.assertEqualXML(r.require('ddd'), '<ddd/>')
            self.assertEqualXML(r.require('bbb'), '<bbb/>')
            with self.assertRaisesRegexp(KeyError, 'parent.*does not have required.*xxx'):
                r.require('xxx')
            r.mark_read('aaa')


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())

