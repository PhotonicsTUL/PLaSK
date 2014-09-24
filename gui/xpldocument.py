from lxml import etree
from controller.geometry.section import GeometryController

from .model.base import SectionModelTreeBased
from .controller.source import SourceEditController
from .controller.defines import DefinesController
from .controller.script import ScriptController
from .controller.multi import GUIAndSourceController
from .controller.connects import ConnectsController
from .controller import materials
from .controller.grids.section import GridsController
from .utils.xml import XML_parser, OrderedTagReader


class XPLDocument(object):

    SECTION_NAMES = ["defines", "materials", "geometry", "grids", "solvers", "connects", "script"]

    def __init__(self, window, filename=None):
        self.window = window
        self.defines = GUIAndSourceController(DefinesController(self))
        self.materials = GUIAndSourceController(materials.MaterialsController(self))
        #self.geometry = GUIAndSourceController(GeometryController(self))
        self.geometry = SourceEditController(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[2]))

        # geometry
        self.grids = GUIAndSourceController(GridsController(self))
        self.controllers = [
            self.defines,
            self.materials,
            self.geometry,
            self.grids,
            SourceEditController(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[4])),  # solvers
            GUIAndSourceController(ConnectsController(self)),   # connects
            ScriptController(self)   # script
        ]
        for c in self.controllers:
            c.model.changed.connect(self.on_model_change)
        self.filename = None
        self.set_changed(False)
        if filename: self.load_from_file(filename)
        #self.tree = etree()

    def on_model_change(self, model, *args, **kwargs):
        """Slot called by model 'changed' signals when user edits any section model"""
        self.set_changed(True)

    def set_changed(self, changed=True):
        self.window.set_changed(changed)

    def load_from_file(self, filename):
        tree = etree.parse(filename, XML_parser)
        with OrderedTagReader(tree.getroot()) as r:
            for i in range(len(XPLDocument.SECTION_NAMES)):
                element = r.get(XPLDocument.SECTION_NAMES[i])
                if element is not None:
                    self.model_by_index(i).set_file_XML_element(element, filename)
                else:
                    self.model_by_index(i).clear()
        self.filename = filename
        self.set_changed(False)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('<plask>\n\n')
            current_line_nr_in_file = 3
            for c in self.controllers:
                c.model.line_nr_in_file = current_line_nr_in_file
                section_string = etree.tostring(c.model.get_file_XML_element(), encoding="UTF-8", pretty_print=True)
                f.write(section_string)
                f.write('\n')
                current_line_nr_in_file += section_string.count('\n') + 1
            f.write('</plask>')
        self.filename = filename
        self.set_changed(False)

    def controller_by_index(self, index):
        return self.controllers[index]

    def controller_by_name(self, section_name):
        return self.controllers[XPLDocument.SECTION_NAMES.index(section_name)]

    def model_by_index(self, index):
        return self.controller_by_index(index).model

    def model_by_name(self, section_name):
        return self.controller_by_name(section_name).model

    def get_info(self, level=None):
        """Get messages from all models, on given level (all by default)."""
        res = []
        for c in self.controllers: res.extend(c.model.get_info(level))
        return res

    def stubs(self):
        res = ""
        for c in self.controllers:
            res += c.model.stubs()
            res += '\n'
        return res
