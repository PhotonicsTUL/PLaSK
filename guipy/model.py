from xml.etree import ElementTree
from models.base import SectionModelTreeBased
from models.defines import DefinesModel
from models.script import ScriptModel

class Model(object):
    
    NAMES = ["defines", "materials", "geometry", "grids", "solvers", "connects", "script"]
     
    def __init__(self):
        object.__init__(self)
        self.models = [
              DefinesModel(),  #defines
              SectionModelTreeBased(Model.NAMES[1]),  #materials
              SectionModelTreeBased(Model.NAMES[2]),  #geometry
              SectionModelTreeBased(Model.NAMES[3]),  #grids
              SectionModelTreeBased(Model.NAMES[4]),  #solvers
              SectionModelTreeBased(Model.NAMES[5]),  #connects
              ScriptModel()   #script
              ]
        #self.tree = ElementTree()
           
    def loadFromFile(self, fileName):
        tree = ElementTree.parse(fileName)
        for i in range(len(Model.NAMES)):
            self.models[i].setXMLElement(tree.getroot().find(Model.NAMES[i]))
        
    def saveToFile(self, fileName):
        root = ElementTree.Element("plask")
        for m in self.models:
            root.append(m.getXMLElement())
        ElementTree.ElementTree(root).write(fileName, encoding="UTF-8") #, encoding, xml_declaration, default_namespace, method)
        
    def getModelByIndex(self, index):
        return self.models[index]
        
    def getModelByName(self, sectionName):
        return self.models[Model.NAMES.index(sectionName)]