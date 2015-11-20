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
from collections import OrderedDict
from lxml import etree
from xml.sax.saxutils import quoteattr

from ...qt import QtCore
from utils import require_str_first_attr_path_component

from ...utils.xml import print_interior, XML_parser, AttributeReader
from ...controller.source import SourceEditController
from .. import TreeFragmentModel, Info, SectionModel
from ..table import TableModel


class Grid(TreeFragmentModel):
    """Base class for models of grids (meshes or generators)"""

    @staticmethod
    def contruct_empty_xml_element(name, type, method=None):
        if method is not None:
            return etree.Element("generator", {"name": name, "type": type, "method": method})
        else:
            return etree.Element("mesh", {"name": name, "type": type})

    def __init__(self, grids_model, name=None, type=None, method=None):
        super(Grid, self).__init__(parent=grids_model)
        if name is not None: self.name = name
        if type is not None: self.type = type
        if method is not None: self._method = method

    def get_xml_element(self):
        return Grid.contruct_empty_xml_element(self.name, self.type, self.method)

    def set_xml_element(self, element):
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
        self.set_xml_element(etree.fromstringlist(tab, parser=XML_parser))   # .encode('utf-8') wymagane (tylko) przez lxml

    @property
    def type_and_kind_str(self):
        from .types import display_name
        if self.is_generator:
            return "%s generator (%s)" % (display_name(self.type), display_name(self.method))
        else:
            return "%s mesh" % display_name(self.type)

    def get_controller(self, document):
        return SourceEditController(document=document, model=self, line_numbers=False)

    @property
    def undo_stack(self):
        return self.tree_parent.undo_stack

    def _append_info(self, res, text, level=None, **kwargs):
        res.append(Info(text, level, **kwargs))

    def _append_error(self, res, text, **kwargs):
        self._append_info(res, text, Info.ERROR, **kwargs)

    def _required(self, res, rows, property, display_name=None, type=None, **kwargs):
        if display_name is None: display_name = '"{}"'.format(require_str_first_attr_path_component(property))
        if type is not None: display_name = 'valid {} value for {}'.format(type, display_name)
        self._append_error(res, 'Specifying {} is required in {} "{}"'.format(display_name, self.type_and_kind_str, self.name),
                           rows=rows, property_name=property, **kwargs)

    def create_info(self, res, rows):
        pass
    #    if not self.object: self._require(res, 'object')

#class Generator(Grid):
#    """Base class for models of generators"""

#class Mesh(Grid):
#    """Base class for models of meshes"""


class TreeFragmentGrid(Grid):
    """Universal grid model, used for grids not supported in other way (data is stored as XML element)"""

    @staticmethod
    def from_xml(grids_model, element):
        return TreeFragmentGrid(grids_model, element=element)

    def __init__(self, grids_model, name=None, type=None, method=None, element=None):
        """Either element or rest of parameters (method is still optional), should be provided."""
        super(TreeFragmentGrid, self).__init__(grids_model)
        if element is None:
            self.element = Grid.contruct_empty_xml_element(name, type, method)
        else:
            self.element = element
        #Grid.__init__(self, name, type, method)

    def set_xml_element(self, element):
        self.element = element
        with AttributeReader(element) as a:
            a.mark_read('name', 'type')
            if self.is_generator: a.mark_read('method')
    #    self.fireChanged()    #TODO ???

    def get_xml_element(self):
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
    def from_xml(grids_model, element):
        return GridWithoutConf(grids_model, element.attrib['name'], element.attrib['type'], element.attrib.get('method', None))

    #def __init__(self, grids_model, name, type, method):
    #    super(GridWithoutConf, self).__init__(grids_model, name, type, method)

    #def get_xml_element(self):
    #    return super(GridWithoutConf, self).get_xml_element()

    #def set_xml_element(self, element):
    #    super(GridWithoutConf, self).set_xml_element()

    def get_controller(self, document):
        from ...controller import NoConfController
        return NoConfController(self.type_and_kind_str + ' has no configuration.')


from ...controller.grids.new_dialog import construct_grid_using_dialog
from .types import construct_grid, display_name


class GridsModel(TableModel):

    def __init__(self, parent=None, info_cb=None, *args):
        super(GridsModel, self).__init__('grids', parent, info_cb, *args)

    def set_xml_element(self, element, undoable=True):
        self._set_entries([] if element is None else [construct_grid(self, g) for g in element], undoable)

    def get_xml_element(self):
        res = etree.Element(self.name)
        for e in self.entries: res.append(e.get_xml_element())
        return res

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2    # 3 if comment supported

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'Name'
            if col == 1: return 'Type (Method)'
            if col == 2: return 'Comment'
        return None

    def get(self, col, row):
        if col == 0: return self.entries[row].name
        if col == 1: return self.entries[row].type_and_kind_str
        if col == 2: return self.entries[row].comment
        raise IndexError('column number for GridsModel should be 0, 1, or 2, but is %d' % col)

    def set(self, col, row, value):
        if col == 0: self.entries[row].name = value
        #else: raise IndexError('column number for MaterialsModel should be 0, 1, or 2, but is %d' % col)

    def flags(self, index):
        flags = super(GridsModel, self).flags(index)
        if index.column() == 1: flags &= ~QtCore.Qt.ItemIsEditable
        return flags

    def create_default_entry(self):
        return construct_grid_using_dialog(self)

    def stubs(self):
        res = 'class MSH(object):\n    """PLaSK object containing the defined meshes."""\n'
        res += '\n'.join("    {0} = mesh.{1}()".format(e.name.replace('-', '_'), display_name(e.type))
                         for e in self.entries if not e.is_generator) + '\n'
        res += 'class MSG(object):\n    """PLaSK object containing the defined mesh generators."""\n'
        res += '\n'.join("    {0} = mesh.{1}.{2}Generator()"
                            .format(e.name.replace('-', '_'), display_name(e.type), display_name(e.method))
                         for e in self.entries if e.is_generator)
        return res

    def create_info(self):
        res = super(GridsModel, self).create_info()
        names = OrderedDict()
        for i, entry in enumerate(self.entries):
            if not entry.name:
                res.append(Info('Grid name is required [row: {}]'.format(i+1), Info.ERROR, rows=(i,), cols=(0,)))
            else:
                names.setdefault(entry.name, []).append(i)
            entry.create_info(res, (i,))
        for name, indexes in names.items():
            if len(indexes) > 1:
                res.append(Info('Duplicated grid name "{}" [rows: {}]'.format(name, ', '.join(map(str, indexes))),
                                Info.ERROR, cols=[0], rows=indexes))
        return res

