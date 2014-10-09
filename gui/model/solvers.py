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

SOLVERS = {}

CATEGORIES = (
    'thermal',
    'electrical',
    'gain',
    'optical'
)

from ..qt import QtCore

from lxml import etree
from xml.sax.saxutils import quoteattr

from ..utils.xml import print_interior, XML_parser, AttributeReader
from ..controller.source import SourceEditController
from .table import TableModel
from . import TreeFragmentModel


class SolverModel(TreeFragmentModel):
    """Base class for all solver models"""

    def __init__(self, category, solver='', name='', parent=None, info_cb=None):
        super(SolverModel, self).__init__(parent, info_cb)
        self.category = category
        self.solver = solver
        self.name = name

    def get_xml_element(self):
        return etree.Element(self.category, {"name": self.name, "solver": self.solver})

    def set_xml_element(self, element):
        self.category = element.tag
        with AttributeReader(element) as attr:
            self.name = attr.get('name', None)
            self.solver = attr.get('solver', None)
            attr.mark_read('lib')

    def set_text(self, text):
        tab = ['<', self.category.encode('utf-8'),
               ' solver=', quoteattr(self.solver).encode('utf-8'),
               ' name=', quoteattr(self.name).encode('utf-8'), '>',
               text.encode('utf-8'),
               '</'. self.category.encode('utf-8'), '>']
        #print ''.join(tab)
        self.set_xml_element(etree.fromstringlist(tab, parser=XML_parser))

    def get_controller(self, document):
        return SourceEditController(document=document, model=self, line_numbers=False)


class SolverModelXML(SolverModel):
    """Universal solver model, used for solvers not supported in other way (data is stored as XML element)"""

    @staticmethod
    def from_xml(element, parent):
        return SolverModelXML(element, parent=parent)

    def __init__(self, element, parent=None, category=None, solver=None, name=None):
        """Either element or rest of parameters (method is still optional), should be provided."""
        if element is None:
            super(SolverModelXML, self).__init__(category, solver, name, parent)
            self.element = SolverModel.get_xml_element(self)
        else:
            super(SolverModelXML, self).__init__(None, parent=parent)
            super(SolverModelXML, self).set_xml_element(element)
            self.element = element

    def set_xml_element(self, element):
        self.element = element
        super(SolverModelXML, self).set_xml_element(element)
    #    self.fireChanged()    #TODO ???

    def get_xml_element(self):
        return self.element

    def get_text(self):
        return print_interior(self.element)


class SolversModel(TableModel):

    def __init__(self, parent=None, info_cb=None, *args):
        super(SolversModel, self).__init__('solvers', parent, info_cb, *args)

    def construct_solver(self, element):
        # TODO
        return SolverModelXML.from_xml(element, self)

    def set_xml_element(self, element):
        self.layoutAboutToBeChanged.emit()
        if element is not None:
            self.entries = [self.construct_solver(item) for item in element]
        else:
            self.entries = []
        self.layoutChanged.emit()
        self.fire_changed()

    def get_xml_element(self):
        res = etree.Element(self.name)
        for e in self.entries:
            res.append(e.get_xml_element())
        return res

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 3

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'Category'
            if col == 1: return 'Solver'
            if col == 2: return 'Name'
        return None

    def get(self, col, row):
        if col == 0: return self.entries[row].category
        if col == 1: return self.entries[row].solver
        if col == 2: return self.entries[row].name
        raise IndexError('column number for SolversModel should be 0, 1, or 2, but is %d' % col)

    def set(self, col, row, value):
        if col == 0: self.entries[row].name = value

    def flags(self, index):
        flags = super(SolversModel, self).flags(index)
        if index.column() < 2: flags &= ~QtCore.Qt.ItemIsEditable
        return flags

    def create_default_entry(self):
        # TODO
        return SolverModel(CATEGORIES[-1], parent=self)
