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
from ..controller.solvers import ConfSolverController
from .table import TableModel
from . import TreeFragmentModel


class Solver(TreeFragmentModel):
    """Base class for all solver models"""

    def __init__(self, category, solver='', name='', parent=None, info_cb=None):
        super(Solver, self).__init__(parent, info_cb)
        self.category = category
        self.solver = solver
        self.name = name
        self.lib = None
        self.geometry = ''
        self.mesh = ''

    def get_xml_element(self):
        return etree.Element(self.category, {"name": self.name, "solver": self.solver})

    def set_xml_element(self, element):
        self.category = element.tag
        with AttributeReader(element) as attr:
            self.name = attr.get('name', None)
            self.solver = attr.get('solver', None)
            self.lib = attr.get('lib', None)

    def set_text(self, text):
        tab = ['<', self.category.encode('utf-8'),
               ' solver=', quoteattr(self.solver).encode('utf-8')]
        if self.lib is not None:
            tab.extend([' lib=', quoteattr(self.lib).encode('utf-8')])
        tab.extend([' name=', quoteattr(self.name).encode('utf-8'), '>',
                    text.encode('utf-8'),
                    '</', self.category.encode('utf-8'), '>'])
        self.set_xml_element(etree.fromstringlist(tab, parser=XML_parser))

    def get_controller(self, document):
        return SourceEditController(document=document, model=self, line_numbers=False)


class TreeFragmentSolver(Solver):
    """Universal solver model, used for solvers not supported in other way (data is stored as XML element)"""

    @staticmethod
    def create_empty(category, solver='', name='', parent=None, info_cb=None):
        element = etree.Element(category, {"name": name, "solver": solver})
        return TreeFragmentSolver(element, parent, info_cb)

    def __init__(self, element, parent=None, info_cb=None):
        """Either element or rest of parameters (method is still optional), should be provided."""
        TreeFragmentModel.__init__(self, parent, info_cb)
        self.element = element

    def set_xml_element(self, element):
        self.element = element
    #    self.fireChanged()    #TODO ???

    def get_xml_element(self):
        return self.element

    def get_text(self):
        return print_interior(self.element)

    @property
    def category(self):
        return self.element.tag
    @category.setter
    def category(self, value):
        self.element.tag = value

    @property
    def lib(self):
        return self.element.attrib.get('lib', None)
    @lib.setter
    def lib(self, value):
        self.element.attrib['lib'] = value

    @property
    def solver(self):
        return self.element.attrib.get('solver', None)
    @solver.setter
    def solver(self, value):
        self.element.attrib['solver'] = value

    @property
    def name(self):
        return self.element.attrib.get('name', None)
    @name.setter
    def name(self, value):
        self.element.attrib['name'] = value


class ConfSolver(Solver):
    """Model for solver with its configuration specified in a simple Python dictionary
       and automatically generated controller widget"""

    def __init__(self, config, category, solver='', name='', parent=None, info_cb=None):
        super(ConfSolver, self).__init__(category, solver, name, parent, info_cb)
        self.config = config
        try:
            self.lib = config['lib']
        except KeyError:
            pass
        self.data = dict((tag, dict((a[0], '') for a in attrs)) for (tag,_,attrs) in self.config['conf'])

    def get_xml_element(self):
        element = etree.Element(self.category, {'name': self.name, 'solver': self.solver})
        if self.lib is not None:
            element.attrib['lib'] = self.lib
        if self.geometry:
            etree.SubElement(element, 'geometry', {'ref': self.geometry})
        if self.mesh:
            etree.SubElement(element, 'mesh', {'ref': self.mesh})
        for tag,_,_ in self.config['conf']:
            attrs = dict((item for item in self.data[tag].items() if item[1]))
            if attrs:
                etree.SubElement(element, tag, attrs)
        return element

    def set_xml_element(self, element):
        super(ConfSolver, self).set_xml_element(element)
        for el in element:
            if el.tag == 'geometry':
                self.geometry = el.attrib.get('ref')
                #TODO report missing or missmatching geometry
            elif el.tag == 'mesh':
                self.mesh = el.attrib.get('ref')
                #TODO report missing or missmatching mesh
            else:
                data = self.data[el.tag]
                with AttributeReader(el) as attr:
                    for name in data:
                        data[name] = attr.get(name, '')

    def get_controller(self, document):
        return ConfSolverController(document, self)


class ConfSolverFactory(object):

    def __init__(self, category, solver, config):
        self.category = category
        self.solver = solver
        self.config = config

    def __call__(self, name='', parent=None, info_cb=None, element=None):
        result = ConfSolver(self.config, self.category, self.solver, name, parent, info_cb)
        if element is not None:
            result.set_xml_element(element)
        return result


class SolversModel(TableModel):

    def __init__(self, parent=None, info_cb=None, *args):
        super(SolversModel, self).__init__('solvers', parent, info_cb, *args)

    def construct_solver(self, element):
        try:
            factory = SOLVERS[element.tag, element.attrib['solver']]
        except KeyError:
            return TreeFragmentSolver(element, self)
        else:
            return factory(element=element, parent=self)


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
        from ..controller.solvers import get_new_solver
        new_solver = get_new_solver()
        if new_solver is not None:
            return TreeFragmentSolver.create_empty(parent=self, **new_solver)
