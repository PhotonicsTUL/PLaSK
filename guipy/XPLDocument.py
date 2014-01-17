from xml.etree import ElementTree
from model.base import SectionModelTreeBased
from controler.base import SourceEditControler
from controler.defines import DefinesControler
from controler.script import ScriptControler

class XPLDocument(object):
    
    SECTION_NAMES = ["defines", "materials", "geometry", "grids", "solvers", "connects", "script"]
     
    def __init__(self, main):
        object.__init__(self)
        self.controlers = [
              DefinesControler(main),  #defines
              SourceEditControler(SectionModelTreeBased(XPLDocument.SECTION_NAMES[1])),  #materials
              SourceEditControler(SectionModelTreeBased(XPLDocument.SECTION_NAMES[2])),  #geometry
              SourceEditControler(SectionModelTreeBased(XPLDocument.SECTION_NAMES[3])),  #grids
              SourceEditControler(SectionModelTreeBased(XPLDocument.SECTION_NAMES[4])),  #solvers
              SourceEditControler(SectionModelTreeBased(XPLDocument.SECTION_NAMES[5])),  #connects
              ScriptControler()   #script
              ]
        #self.tree = ElementTree()
           
    def loadFromFile(self, fileName):
        tree = ElementTree.parse(fileName)
        for i in range(len(XPLDocument.SECTION_NAMES)):
            element = tree.getroot().find(XPLDocument.SECTION_NAMES[i])
            if isinstance(element, ElementTree.Element):
                self.getModelByIndex(i).setFileXMLElement(element)
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