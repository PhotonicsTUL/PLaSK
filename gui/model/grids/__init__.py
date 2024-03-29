# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


# Base classes for entries in grids model
from lxml import etree
from xml.sax.saxutils import quoteattr

from ...qt.QtCore import *
from ...utils import require_str_first_attr_path_component
from ...utils.xml import XMLparser, AttributeReader, OrderedTagReader
from ...controller.source import SourceEditController
from .. import TreeFragmentModel, Info
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
        super().__init__(parent=grids_model)
        if name is not None: self.name = name
        if type is not None: self.type = type
        if method is not None: self._method = method
        self.comments = []
        self.endcomments = []

    def make_xml_element(self):
        return Grid.contruct_empty_xml_element(self.name, self.type, self.method)

    def load_xml_element(self, element):
        with AttributeReader(element) as a:
            self.name = a.get('name', None)
            a.mark_read('type')
            if self.is_generator: a.mark_read('method')

    def save_endcomments(self, res):
        for c in self.endcomments:
            res.append(etree.Comment(c))

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
            tab = ['<generator name=', quoteattr(self.name), ' type=',
                   quoteattr(self.type), ' method=', quoteattr(self.method), '>',
                   text, '</generator>']
        else:
            tab = ['<mesh name=', quoteattr(self.name), ' type=',
                   quoteattr(self.type), '>', text, '</mesh>']
        #print ''.join(tab)
        self.load_xml_element(etree.fromstringlist(tab, parser=XMLparser))

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
        res.append(Info('{} in {} "{}"'.format(text, self.type_and_kind_str, self.name), level, **kwargs))

    def _append_error(self, res, text, **kwargs):
        self._append_info(res, text, Info.ERROR, **kwargs)

    def _required(self, res, rows, property, display_name=None, type=None, **kwargs):
        if display_name is None: display_name = '"{}"'.format(require_str_first_attr_path_component(property))
        if type is not None: display_name = 'valid {} value for {}'.format(type, display_name)
        self._append_error(res, 'Specifying {} is required'.format(display_name),
                           rows=rows, property=property, **kwargs)

    def create_info(self, res, rows):
        pass
    #    if not self.object: self._require(res, 'object')


class TreeFragmentGrid(Grid):
    """Universal grid model, used for grids not supported in other way (data is stored as XML element)"""

    @staticmethod
    def from_xml(grids_model, element):
        return TreeFragmentGrid(grids_model, element=element)

    def __init__(self, grids_model, name=None, type=None, method=None, element=None):
        """Either element or rest of parameters (method is still optional), should be provided."""
        super().__init__(grids_model)
        if element is None:
            self.element = Grid.contruct_empty_xml_element(name, type, method)
        else:
            self.element = element
        #Grid.__init__(self, name, type, method)

    def load_xml_element(self, element):
        self.element = element
        with AttributeReader(element) as a:
            a.mark_read('name', 'type')
            if self.is_generator: a.mark_read('method')
    #    self.fireChanged()    #TODO ???

    def make_xml_element(self):
        return self.element.get_etree_element()

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
        return GridWithoutConf(grids_model,
                               element.attrib['name'],
                               element.attrib['type'],
                               element.attrib.get('method', None))

    def get_controller(self, document):
        from ...controller import NoConfController
        return NoConfController(self.type_and_kind_str + ' has no configuration.', document=document, model=self)


from ...controller.grids.new_dialog import construct_grid_using_dialog
from .types import construct_grid, display_name


class GridsModel(TableModel):

    def __init__(self, parent=None, info_cb=None, *args):
        super().__init__('grids', parent, info_cb, *args)
        self._messages = []
        self.endcomments = []

    def load_xml_element(self, element, undoable=True):
        with OrderedTagReader(element) as reader:
            self._set_entries([] if element is None else [construct_grid(self, g) for g in reader], undoable)
            self.endcomments = reader.get_comments()

    def make_xml_element(self):
        res = etree.Element(self.name)
        for e in self.entries:
            for c in e.comments:
                res.append(etree.Comment(c))
            res.append(e.make_xml_element())
        for c in self.endcomments:
            res.append(etree.Comment(c))
        return res

    def columnCount(self, parent=QModelIndex()):
        return 2

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if col == 0: return 'Name'
            if col == 1: return 'Type (Method)'
        return None

    def get(self, col, row):
        if col == 0: return self.entries[row].name
        if col == 1: return self.entries[row].type_and_kind_str
        raise IndexError('column number for GridsModel should be 0 or 1, but is %d' % col)

    def set(self, col, row, value):
        if col == 0: self.entries[row].name = value
        #else: raise IndexError('column number for MaterialsModel should be 0, 1, or 2, but is %d' % col)

    def flags(self, index):
        flags = super().flags(index)
        if index.column() == 1: flags &= ~Qt.ItemFlag.ItemIsEditable
        return flags

    def create_default_entry(self):
        return construct_grid_using_dialog(self)

    def stubs(self):
        res = 'class MSH:\n    """PLaSK object containing the defined meshes."""\n'
        res += '\n'.join("    {0} = mesh.{1}()".format(e.name.replace('-', '_'), display_name(e.type))
                         for e in self.entries if not e.is_generator) + '\n'
        res += '\n'.join("    {0} = mesh.{1}.{2}Generator()"
                            .format(e.name.replace('-', '_'), display_name(e.type), display_name(e.method))
                         for e in self.entries if e.is_generator)
        return res

    def create_info(self):
        res = super().create_info()
        names = {}
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
        pos = 0
        for msg in self._messages:
            if msg['level'] >= Info.WARNING:
                res.insert(pos, Info(**msg))
                pos += 1
            else:
                res.append(Info(**msg))
        return res

    def add_info_message(self, msg=None, level=Info.INFO, **kwargs):
        if isinstance(msg, tuple):
            info = dict(text=msg[0], level=level, line=msg[1])
        else:
            info = dict(text=msg, level=level)
        info.update(kwargs)
        self._messages.append(info)

    def clear_info_messages(self):
        self._messages = []
