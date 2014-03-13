# Base classes for entries in grids model
from lxml.etree import ElementTree
from model.info import InfoSource
from utils.xml import print_interior
from xml.sax.saxutils import quoteattr

class Grid(InfoSource): # or (TreeFragmentModel)??
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
        
    def setText(self, text):
        if self.is_generator():
            tab = ['<generator name="', quoteattr(self.name), '" type="', quoteattr(self.type), '" method="', quoteattr(self.method), '">', text.encode('utf-8'), '</generator>' ]
        else:
            tab = ['<mesh name="', quoteattr(self.name), '" type="', quoteattr(self.type), '">', text.encode('utf-8'), '</mesh>' ]
        self.setXMLElement(ElementTree.fromstringlist(tab))   # .encode('utf-8') wymagane (tylko) przez lxml
        
    @property
    def type_and_kind_str(self):
        if self.is_generator:
            return "%s generator (%s)" % (self.type, self.method)
        else:
            return "%s mesh" % self.type
        
#class Generator(Grid):
#    """Base class for models of generators"""

#class Mesh(Grid):
#    """Base class for models of meshes"""

class GridTreeBased(Grid):
    """Universal grid model, used for not supported grids (data are stored as XML element)"""

    @staticmethod
    def from_XML(element):
        e = GridTreeBased(element.attrib['name'], element.attrib['type'], element.attrib['method'] if element.tag == 'generator' else None)
        e.setXMLElement(element)
        return e

    def __init__(self, name, type, method = None):
        Grid.__init__(self, name, type, method)

    def setXMLElement(self, element):
        self.element = element
    #    self.fireChanged()    #TODO ???

    def getXMLElement(self):
        return self.element
  
    def getText(self):
        return print_interior(self.getXMLElement())
