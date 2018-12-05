from lxml.etree import SubElement

from . import Grid
from ...utils.validators import can_be_float


class TriangularTriangleGenerator(Grid):

    @staticmethod
    def from_xml(grids_model, element):
        e = TriangularTriangleGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.set_xml_element(element)
        return e

    def __init__(self, grids_model, name, type, method='triangle'):
        super(TriangularTriangleGenerator, self).__init__(grids_model, name, type, method)
        self.maxarea = None
        self.minangle = None

    @property
    def dim(self):
        return 2

    def get_xml_element(self):
        res = super(TriangularTriangleGenerator, self).get_xml_element()
        options_dict = {}
        if self.maxarea is not None: options_dict['maxarea'] = self.maxarea
        if self.minangle is not None: options_dict['minangle'] = self.minangle
        if len(options_dict) > 0:
            SubElement(res, "options", attrib=options_dict)
        return res

    def set_xml_element(self, element):
        super(TriangularTriangleGenerator, self).set_xml_element(element)
        options = element.find('options')
        if options is not None:
            self.maxarea = options.attrib.get('maxarea')
            self.minangle = options.attrib.get('minangle')

    def get_controller(self, document):
        from ...controller.grids.generator_triangular import TriangularTriangleGeneratorController
        return TriangularTriangleGeneratorController(document=document, model=self)

    def create_info(self, res, rows):
        super(TriangularTriangleGenerator, self).create_info(res, rows)
        if not can_be_float(self.maxarea):
            self._required(res, rows, 'maxarea', type='float')
        if not can_be_float(self.minangle):
            self._required(res, rows, 'minangle', type='float')
