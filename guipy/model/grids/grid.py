# Base classes for entries in grids model
from lxml.etree import ElementTree
from model.info import InfoSource

class Grid(InfoSource):
    """Base class for models of grids (meshes or generators)"""
    
    def __init__(self, name, type, method = None):
        object.__init__(self)
        self.name = name
        self.type = type
        if method != None:
            self.__is_generator__ = True
            self.method = method
        else:
            self.__is_generator__ = False
    
    def getXMLElement(self):
        if self.is_generator():
            return ElementTree.Element("generator", { "name": self.name, "type": self.type, "method": self.method })
        else:
            return ElementTree.Element("mesh", { "name": self.name, "type": self.type })
        
    @property
    def is_generator(self):
        return self.__is_generator__
    
    @property
    def is_mesh(self):
        return not self.__is_generator__
        
#class Generator(Grid):
#    """Base class for models of generators"""

#class Mesh(Grid):
#    """Base class for models of meshes"""

class GridTreeBased(Grid):
    """Universal grid model, used for not supported grids (data are stored as XML element)"""

    def __init__(self, name, type, method = None):
        Grid.__init__(self, name, type, method)
        #self.element = Grid.getXMLElement(self)

    def setXMLElement(self, element):
        self.element = element
    #    self.fireChanged()    #TODO class for InfoSource + change signal = TreeFragmentModel

    def getXMLElement(self):
        return self.element
  