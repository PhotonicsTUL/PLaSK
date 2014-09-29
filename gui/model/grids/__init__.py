# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# Base classes for entries in grids model
from lxml import etree
from xml.sax.saxutils import quoteattr

from .. import TreeFragmentModel
from ...utils.xml import print_interior, XML_parser, AttributeReader
from ..info import InfoSource


class Grid(TreeFragmentModel):
    """Base class for models of grids (meshes or generators)"""

    @staticmethod
    def contruct_empty_XML_element(name, type, method=None):
        if method is not None:
            return etree.Element("generator", {"name": name, "type": type, "method": method})
        else:
            return etree.Element("mesh", {"name": name, "type": type})

    def __init__(self, grids_model, name=None, type=None, method=None):
        super(Grid, self).__init__(parent=grids_model)
        if name is not None: self.name = name
        if type is not None: self.type = type
        if method is not None: self._method = method

    def get_XML_element(self):
        return Grid.contruct_empty_XML_element(self.name, self.type, self.method)

    def set_XML_element(self, element):
        with AttributeReader(element) as a:
            self.name = a.get('name', None)
            a.mark_read('type')
            if self.is_generator: a.mark_read('method')

    @property
    def method(self):
        return getattr(self, '_method', None)

    @property
    def is_generator(self):
        return self.method is not None

    @property
    def is_mesh(self):
        return self.method is None

    def set_text(self, text):
        if self.is_generator:
            tab = ['<generator name=', quoteattr(self.name).encode('utf-8'), ' type=',
                   quoteattr(self.type).encode('utf-8'), ' method=', quoteattr(self.method).encode('utf-8'), '>',
                   text.encode('utf-8'), '</generator>']
        else:
            tab = ['<mesh name=', quoteattr(self.name).encode('utf-8'), ' type=',
                   quoteattr(self.type).encode('utf-8'), '>', text.encode('utf-8'), '</mesh>']
        #print ''.join(tab)
        self.set_XML_element(etree.fromstringlist(tab, parser=XML_parser))   # .encode('utf-8') wymagane (tylko) przez lxml

    @property
    def type_and_kind_str(self):
        from .types import display_name
        if self.is_generator:
            return "%s generator (%s)" % (display_name(self.type), display_name(self.method))
        else:
            return "%s mesh" % display_name(self.type)

    def get_controller(self, document):
        from ...controller.source import SourceEditController
        return SourceEditController(document=document, model=self)

#class Generator(Grid):
#    """Base class for models of generators"""

#class Mesh(Grid):
#    """Base class for models of meshes"""


class GridTreeBased(Grid):
    """Universal grid model, used for grids not supported in other way (data are stored as XML element)"""

    @staticmethod
    def from_XML(grids_model, element):
        return GridTreeBased(grids_model, element=element)

    def __init__(self, grids_model, name=None, type=None, method=None, element=None):
        """Either element or rest of parameters (method is still optional), should be provided."""
        super(GridTreeBased, self).__init__(grids_model)
        if element is None:
            self.element = Grid.contruct_empty_XML_element(name, type, method)
        else:
            self.element = element
        #Grid.__init__(self, name, type, method)

    def set_XML_element(self, element):
        self.element = element
        with AttributeReader(element) as a:
            a.mark_read('name', 'type')
            if self.is_generator: a.mark_read('method')
    #    self.fireChanged()    #TODO ???

    def get_XML_element(self):
        return self.element

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


class GridWithoutConf(Grid):
    """Model for all grids that does not require any configuration."""

    @staticmethod
    def from_XML(grids_model, element):
        return GridWithoutConf(grids_model, element.attrib['name'], element.attrib['type'], element.attrib.get('method', None))

    #def __init__(self, grids_model, name, type, method):
    #    super(GridWithoutConf, self).__init__(grids_model, name, type, method)

    #def get_XML_element(self):
    #    return super(GridWithoutConf, self).get_XML_element()

    #def set_XML_element(self, element):
    #    super(GridWithoutConf, self).set_XML_element()

    def get_controller(self, document):
        from ...controller.base import NoConfController
        return NoConfController(self.type_and_kind_str + ' has no configuration.')