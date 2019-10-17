import sys, os
import unittest
sys.path.insert(0, os.path.abspath('../..'))

from gui_test_utils import GUITestCase
from gui.model.solvers import FilterSolver

from lxml import etree

class TestGUIModelSolversFilter(GUITestCase):

    def setUp(self):
        self.filter = FilterSolver('Temperature', 'filter_name')
        self.filter.geometry = 'geometry_name'

    def test_get_xml_element(self):
        self.assertEqualXML(self.filter.get_xml_element(),
            '<filter for="Temperature" name="filter_name" geometry="geometry_name"/>')

    def test_set_xml_element(self):
        self.filter.set_xml_element(etree.XML(
            '''<filter for="Luminescence" name="name_from_XML" geometry="geometry_from_XML">
                <!-- to ignore --></filter>'''
        ))
        self.assertEqual(self.filter.what, 'Luminescence')
        self.assertEqual(self.filter.name, 'name_from_XML')
        self.assertEqual(self.filter.geometry, 'geometry_from_XML')



if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
