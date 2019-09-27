import unittest
from lxml import etree

class GUITestCase(unittest.TestCase):

    # test if two xml.etree.ElementTree.Element are equal
    def assertEqualXMLElements(self, e1, e2):
        self.assertEqual(e1.tag, e2.tag)
        self.assertEqual(e1.text, e2.text)
        self.assertEqual(e1.tail, e2.tail)
        self.assertEqual(e1.attrib, e2.attrib)
        self.assertEqual(len(e1), len(e2))
        for c1, c2 in zip(e1, e2):
            self.assertEqualXMLElements(c1, c2)

    # test if two xml.etree.ElementTree.Element/str are equal
    def assertEqualXML(self, xml1, xml2):
        if isinstance(xml1, str): xml1 = etree.XML(xml1)
        if isinstance(xml2, str): xml2 = etree.XML(xml2)
        self.assertEqualXMLElements(xml1, xml2)