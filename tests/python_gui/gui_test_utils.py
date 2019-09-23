from lxml import etree

# test if two xml.etree.ElementTree.Element are equal
def assert_elements_equal(tester, e1, e2):
    tester.assertEqual(e1.tag, e2.tag)
    tester.assertEqual(e1.text, e2.text)
    tester.assertEqual(e1.tail, e2.tail)
    tester.assertEqual(e1.attrib, e2.attrib)
    tester.assertEqual(len(e1), len(e2))
    for c1, c2 in zip(e1, e2):
        assert_elements_equal(tester, c1, c2)

def assert_XML_equal(tester, xml1, xml2):
    if isinstance(xml1, str): xml1 = etree.XML(xml1)
    if isinstance(xml2, str): xml2 = etree.XML(xml2)
    assert_elements_equal(tester, xml1, xml2)