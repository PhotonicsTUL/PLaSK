from lxml import etree as ElementTree
from model.base import SectionModelTreeBased
from controller.source import SourceEditController
from controller.defines import DefinesController
from controller.script import ScriptController
from controller.multi import GUIAndSourceController
from controller.connects import ConnectsController
from controller import materials
from controller.grids.section import GridsController

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
           
    def loadFromFile(self, fileName):
        tree = ElementTree.parse(fileName)
        for i in range(len(XPLDocument.SECTION_NAMES)):
            element = tree.getroot().find(XPLDocument.SECTION_NAMES[i])
            if element is not None:
                self.getModelByIndex(i).set_file_XML_element(element, fileName)
            else:
                self.getModelByIndex(i).clear()
        
    def saveToFile(self, fileName):
        root = ElementTree.Element("plask")
        for c in self.controllers:
            root.append(c.model.get_file_XML_element())
        ElementTree.ElementTree(root).write(fileName, encoding="UTF-8") #, encoding, xml_declaration, default_namespace, method)
        
    def getControllerByIndex(self, index):
        return self.controllers[index]
        
    def getControllerByName(self, sectionName):
        return self.controllers[XPLDocument.SECTION_NAMES.index(sectionName)]
        
    def getModelByIndex(self, index):
        return self.getControllerByIndex(index).model
        
    def getModelByName(self, sectionName):
        return self.getControllerByName(sectionName).model
    
    def getInfo(self, level = None):
        """Get messages from all models, on given level (all by default)."""
        res = []
        for c in self.controllers: res.extend(c.model.getInfo(level))
        return res
    
    def stubs(self):
        res = ""
        for c in self.controllers:
            res += c.model.stubs()
            res += '\n'
        return res  
