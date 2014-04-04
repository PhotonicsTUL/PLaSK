# Base classes for entries in grids model
from lxml import etree as ElementTree
from xml.sax.saxutils import quoteattr

from ...utils.xml import print_interior, XML_parser
from ..info import InfoSource

class Grid(InfoSource): # or (TreeFragmentModel)??
    """Base class for models of grids (meshes or generators)"""

    @staticmethod
    def contruct_empty_XML_element(name, type, method = None):
        if method != None:
            return ElementTree.Element("generator", { "name": name, "type": type, "method": method })
        else:
            return ElementTree.Element("mesh", { "name": name, "type": type })

    def __init__(self, grids_model, name = None, type = None, method = None):
        object.__init__(self)
        self.model = grids_model
        if name != None: self.name = name
        if type != None: self.type = type
        if method != None: self.__method__ = method

    def get_XML_element(self):
        return Grid.contruct_empty_XML_element(self.name, self.type, self.method)

    @property
    def method(self):
        return getattr(self, '__method__', None)

    @property
    def is_generator(self):
        return self.method != None

    @property
    def is_mesh(self):
        return self.method == None

    def set_text(self, text):
        if self.is_generator:
            tab = ['<generator name=', quoteattr(self.name), ' type=', quoteattr(self.type), ' method=', quoteattr(self.method), '>', text.encode('utf-8'), '</generator>' ]
        else:
            tab = ['<mesh name=', quoteattr(self.name), ' type=', quoteattr(self.type), '>', text.encode('utf-8'), '</mesh>' ]
        #print ''.join(tab)
        self.set_XML_element(ElementTree.fromstringlist(tab, parser = XML_parser))   # .encode('utf-8') wymagane (tylko) przez lxml

    @property
    def type_and_kind_str(self):
        from .types import display_name
        if self.is_generator:
            return "%s generator (%s)" % (display_name(self.type), display_name(self.method))
        else:
            return "%s mesh" % display_name(self.type)

    def is_read_only(self):
        return self.model.is_read_only()

    def get_controller(self):
        from ...controller.source import SourceEditController
        return SourceEditController(model = self)

#class Generator(Grid):
#    """Base class for models of generators"""

#class Mesh(Grid):
#    """Base class for models of meshes"""

class GridTreeBased(Grid):
    """Universal grid model, used for grids not supported in other way (data are stored as XML element)"""

    @staticmethod
    def from_XML(grids_model, element):
        return GridTreeBased(grids_model, element = element)

    def __init__(self, grids_model, name = None, type = None, method = None, element = None):
        """Either element or rest of parameters (method is still optional), should be provided."""
        super(GridTreeBased, self).__init__(grids_model)
        if element == None:
            self.element = Grid.contruct_empty_XML_element(name, type, method)
        else:
            self.element = element
        #Grid.__init__(self, name, type, method)

    def set_XML_element(self, element):
        self.element = element
    #    self.fireChanged()    #TODO ???

    def get_XML_element(self):
        return self.element

    def get_text(self):
        return print_interior(self.get_XML_element())

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
