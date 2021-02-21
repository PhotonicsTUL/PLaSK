from lxml import etree

from . import Grid
from ...utils.xml import UnorderedTagReader
from ...utils.validators import can_be_float, can_be_bool


class TriangularTriangleGenerator(Grid):

    @staticmethod
    def from_xml(grids_model, element):
        e = TriangularTriangleGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.load_xml_element(element)
        return e

    def __init__(self, grids_model, name, type, method='triangle'):
        super().__init__(grids_model, name, type, method)
        self.maxarea = None
        self.minangle = None
        self.full = None
        self.options_comments = []

    @property
    def dim(self):
        return 2

    def make_xml_element(self):
        res = super().make_xml_element()
        options_dict = {}
        if self.maxarea is not None: options_dict['maxarea'] = self.maxarea
        if self.minangle is not None: options_dict['minangle'] = self.minangle
        if self.full is not None: options_dict['full'] = self.full
        for c in self.options_comments:
            res.append(etree.Comment(c))
        if len(options_dict) > 0:
            etree.SubElement(res, "options", attrib=options_dict)
        self.save_endcomments(res)
        return res

    def load_xml_element(self, element):
        super().load_xml_element(element)
        with UnorderedTagReader(element) as reader:
            options = reader.find('options')
            if options is not None:
                self.maxarea = options.attrib.get('maxarea')
                self.minangle = options.attrib.get('minangle')
                self.full = options.attrib.get('full')
                self.options_comments = options.comments
            else:
                self.options_comments = []
            self.endcomments = reader.get_comments()

    def get_controller(self, document):
        from ...controller.grids.generator_triangular import TriangularTriangleGeneratorController
        return TriangularTriangleGeneratorController(document=document, model=self)

    def create_info(self, res, rows):
        super().create_info(res, rows)
        if not can_be_float(self.maxarea):
            self._required(res, rows, 'maxarea', type='float')
        if not can_be_float(self.minangle):
            self._required(res, rows, 'minangle', type='float')
        if not can_be_bool(self.full):
            self._required(res, rows, 'full', type='bool')
