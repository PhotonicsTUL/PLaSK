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

    def __init__(self, main):
        object.__init__(self)
        self.defines = GUIAndSourceController(DefinesController(self))
        self.materials = GUIAndSourceController(materials.MaterialsController(self))
        #geometry
        self.grids = GUIAndSourceController(GridsController(self))
        self.controllers = [
              self.defines,
              self.materials,
              SourceEditController(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[2])),  #geometry
              self.grids,
              SourceEditController(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[4])),  #solvers
              GUIAndSourceController(ConnectsController(self)),   #connects
              ScriptController(self)   #script
              ]
        self.mainWindow = main
        #self.tree = ElementTree()

    def load_from_file(self, fileName):
        tree = ElementTree.parse(fileName, XML_parser)
        for i in range(len(XPLDocument.SECTION_NAMES)):
            element = tree.getroot().find(XPLDocument.SECTION_NAMES[i])
            if element is not None:
                self.model_by_index(i).set_file_XML_element(element, fileName)
            else:
                self.model_by_index(i).clear()

    def save_to_file(self, fileName):
        root = ElementTree.Element("plask")
        for c in self.controllers:
            root.append(c.model.get_file_XML_element())
        ElementTree.ElementTree(root).write(fileName, encoding="UTF-8", pretty_print=True) #, encoding, xml_declaration, default_namespace, method)

    def controller_by_index(self, index):
        return self.controllers[index]

    def controller_by_name(self, sectionName):
        return self.controllers[XPLDocument.SECTION_NAMES.index(sectionName)]

    def model_by_index(self, index):
        return self.controller_by_index(index).model

    def model_by_name(self, sectionName):
        return self.controller_by_name(sectionName).model

    def get_info(self, level = None):
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
