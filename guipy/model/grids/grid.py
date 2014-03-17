# Base classes for entries in grids model
from lxml.etree import ElementTree
from model.info import InfoSource
from utils.xml import print_interior
from xml.sax.saxutils import quoteattr

class Grid(InfoSource): # or (TreeFragmentModel)??
    """Base class for models of grids (meshes or generators)"""

    @staticmethod
    def contruct_empty_XML_element(name, type, method = None):
        if method != None:
            return ElementTree.Element("generator", { "name": name, "type": type, "method": method })
        else:
            return ElementTree.Element("mesh", { "name": name, "type": type })
    
    def __init__(self, name = None, type = None, method = None):
        object.__init__(self)
        if name != None: self.name = name
        if type != None: self.type = type
        if method != None: self.__method__ = method
       
    def getXMLElement(self):
        return Grid.contruct_empty_XML_element(self.name, self.type, self.method)
        
    @property
    def method(self):
        return getattr(self, '__method__', None)
        
    @property
    def is_generator(self):
        return self.method != None
    
    @property
    def is_mesh(self):
        return not self.method == None
        
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
    """Universal grid model, used for grids not supported in other way (data are stored as XML element)"""

    @staticmethod
    def from_XML(element):
        return GridTreeBased(element = element)

    def __init__(self, name = None, type = None, method = None, element = None):
        """Either element or rest of parameters (method is still optional), should be provided."""
        super(GridTreeBased, self).__init__()
        if element == None:
            self.element = Grid.contruct_empty_XML_element(name, type, method)
        else:
            self.element = element
        #Grid.__init__(self, name, type, method)

    def setXMLElement(self, element):
        self.element = element
    #    self.fireChanged()    #TODO ???

    def getXMLElement(self):
        return self.element
  
    def getText(self):
        return print_interior(self.getXMLElement())

    @property
    def method(self):
        return self.element.attrib.get('method', None)
    
    @property
    def name(self):
        return self.element.attrib.get('name', '')
    
    @name.setter
    def name(self, v):
        self.element.attrib['name'] = v
        
    @property
    def type(self):
        return self.element.attrib.get('type', '')  