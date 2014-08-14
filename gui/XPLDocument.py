from lxml import etree as ElementTree

from .model.base import SectionModelTreeBased
from .controller.source import SourceEditController
from .controller.defines import DefinesController
from .controller.script import ScriptController
from .controller.multi import GUIAndSourceController
from .controller.connects import ConnectsController
from .controller import materials
from .controller.grids.section import GridsController
from .utils.xml import XML_parser


class XPLDocument(object):

    SECTION_NAMES = ["defines", "materials", "geometry", "grids", "solvers", "connects", "script"]

    def __init__(self, window):
        self.window = window
        self.defines = GUIAndSourceController(DefinesController(self))
        self.materials = GUIAndSourceController(materials.MaterialsController(self))
        # geometry
        self.grids = GUIAndSourceController(GridsController(self))
        self.controllers = [
            self.defines,
            self.materials,
            SourceEditController(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[2])),  # geometry
            self.grids,
            SourceEditController(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[4])),  # solvers
            GUIAndSourceController(ConnectsController(self)),   # connects
            ScriptController(self)   # script
        ]
        for c in self.controllers:
            c.model.changed.connect(self._on_model_change)
        #self.tree = ElementTree()
        self.set_changed(False)

    def _on_model_change(self, model):
        """Slot called by model 'changed' signals when user edits any section model"""
        self.set_changed()

    def load_from_file(self, filename):
        tree = ElementTree.parse(filename, XML_parser)
        for i in range(len(XPLDocument.SECTION_NAMES)):
            element = tree.getroot().find(XPLDocument.SECTION_NAMES[i])
            if element is not None:
                self.model_by_index(i).set_file_XML_element(element, filename)
            else:
                self.model_by_index(i).clear()
        self.set_changed(False)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('<plask>\n\n')
            for c in self.controllers:
                f.write(ElementTree.tostring(c.model.get_file_XML_element(), encoding="UTF-8", pretty_print=True))
                f.write('\n')
            f.write('</plask>')
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

    def set_changed(self, changed=True):
        """Set changed flags in the document window"""
        self.window.setWindowModified(changed)