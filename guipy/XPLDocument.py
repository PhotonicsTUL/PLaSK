from xml.etree import ElementTree
from model.base import SectionModelTreeBased
from controler.source import SourceEditControler
from controler.defines import DefinesControler
from controler.script import ScriptControler
from controler.multi import GUIAndSourceControler
from controler.connects import ConnectsControler
from model.materials import MaterialsModel

class XPLDocument(object):
    
    SECTION_NAMES = ["defines", "materials", "geometry", "grids", "solvers", "connects", "script"]
     
    def __init__(self, main):
        object.__init__(self)
        self.defines = GUIAndSourceControler(DefinesControler(self))
        self.controlers = [
              self.defines,
              SourceEditControler(self, MaterialsModel()),  #materials
              SourceEditControler(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[2])),  #geometry
              SourceEditControler(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[3])),  #grids
              SourceEditControler(self, SectionModelTreeBased(XPLDocument.SECTION_NAMES[4])),  #solvers
              GUIAndSourceControler(ConnectsControler(self)),   #connects
              ScriptControler(self)   #script
              ]
        self.mainWindow = main
        #self.tree = ElementTree()
           
    def loadFromFile(self, fileName):
        tree = ElementTree.parse(fileName)
        for i in range(len(XPLDocument.SECTION_NAMES)):
            element = tree.getroot().find(XPLDocument.SECTION_NAMES[i])
            if isinstance(element, ElementTree.Element):
                self.getModelByIndex(i).setFileXMLElement(element, fileName)
            else:
                self.getModelByIndex(i).clear()
        
    def saveToFile(self, fileName):
        root = ElementTree.Element("plask")
        for c in self.controlers:
            root.append(c.model.getFileXMLElement())
        ElementTree.ElementTree(root).write(fileName, encoding="UTF-8") #, encoding, xml_declaration, default_namespace, method)
        
    def getControlerByIndex(self, index):
        return self.controlers[index]
        
    def getControlerByName(self, sectionName):
        return self.controlers[XPLDocument.SECTION_NAMES.index(sectionName)]
        
    def getModelByIndex(self, index):
        return self.getControlerByIndex(index).model
        
    def getModelByName(self, sectionName):
        return self.getControlerByName(sectionName).model
    
    def getInfo(self, level = None):
        """Get messages from all models, on given level (all by default)."""
        res = []
        for c in self.controlers: res.extend(c.model.getInfo(level))
        return res