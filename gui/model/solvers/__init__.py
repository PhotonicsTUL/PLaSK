# coding: utf-8
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

import os.path
from collections import OrderedDict
from lxml import etree
from xml.sax.saxutils import quoteattr

from ...qt.QtCore import *
from ...utils.xml import print_interior, XML_parser, AttributeReader
from ...controller.source import SourceEditController
from ..table import TableModel
from .. import TreeFragmentModel, Info

SOLVERS = {}

CATEGORIES = [
    'thermal',
    'electrical',
    'gain',
    'optical',
    None,
    'meta',
    None
]

SUFFIXES = ('2D', 'Cyl', '3D')

def suffix(solver):
    for s in SUFFIXES:
        if solver.endswith(s):
            return s
    return ''


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
            if 'lib' in attr:
                self.lib = attr['lib']

    def set_text(self, text):
        tab = ['<', self.category,
               ' solver=', quoteattr(self.solver)]
        if self.lib is not None:
            tab.extend([' lib=', quoteattr(self.lib)])
        tab.extend([' name=', quoteattr(self.name), '>',
                    text,
                    '</', self.category, '>'])
        self.set_xml_element(etree.fromstringlist(tab, parser=XML_parser))

    def get_controller(self, document):
        return SourceEditController(document=document, model=self, line_numbers=False)

    def stub(self):
        if self.category is not None and self.solver is not None:
            lib = self.lib
            # if lib is None:
            #     try:
            #         prefix = os.path.dirname(os.path.dirname(sys.executable))
            #         lst_re = re.compile(r'(\w+)\.{}'.format(self.solver))
            #         with open(os.path.join(prefix, 'lib', 'plask', 'solvers', self.category, 'solvers.lst')) as lfile:
            #             for line in lfile:
            #                 match = lst_re.match(line)
            #                 if match:
            #                     lib = match.group(1)
            #                     break
            #     except (IOError, SystemError):
            #         pass
            if lib is not None:
                return "import {1}.{2}.{3} as {0}\n{0} = {0}()".format(self.name, self.category, lib, self.solver)
            else:
                return "import {1}.{2} as {0}\n{0} = {0}()".format(self.name, self.category, self.solver)
        else:
            return "{} = None".format(self.name)


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


from .filter import FilterSolver


class SolversModel(TableModel):

    def __init__(self, parent=None, info_cb=None, *args):
        super(SolversModel, self).__init__('solvers', parent, info_cb, *args)
        self.local_categories = []
        self.local_solvers = {}

    def construct_solver(self, element):
        if element.tag == 'filter':
            filter = FilterSolver(parent=self)
            filter.set_xml_element(element)
            return filter
        else:
            try:
                key = element.tag, element.attrib['solver']
                if key in self.local_solvers:
                    factory = self.local_solvers[key]
                else:
                    factory = SOLVERS[key]
            except KeyError:
                return TreeFragmentSolver(element, self)
            else:
                return factory(element=element, parent=self)

    def set_file_xml_element(self, element, filename=None):
        update_solvers(filename, self)
        super(SolversModel, self).set_file_xml_element(element, filename)

    def set_xml_element(self, element, undoable=True):
        self._set_entries([] if element is None else [self.construct_solver(item) for item in element], undoable)

    def get_xml_element(self):
        res = etree.Element(self.name)
        for e in self.entries:
            res.append(e.get_xml_element())
        return res

    def columnCount(self, parent=QModelIndex()):
        return 3

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if col == 0: return 'Category'
            if col == 1: return 'Solver'
            if col == 2: return 'Name'
        return None

    def get(self, col, row):
        if col == 0:
            category = self.entries[row].category
            if category == 'filter': return 'FILTER'
            else: return category.title()
        if col == 1: return self.entries[row].solver
        if col == 2: return self.entries[row].name
        raise IndexError('column number for SolversModel should be 0, 1, or 2, but is %d' % col)

    def set(self, col, row, value):
        if col == 2: self.entries[row].name = value

    def flags(self, index):
        flags = super(SolversModel, self).flags(index)
        if index.column() < 2: flags &= ~Qt.ItemIsEditable
        return flags

    def create_default_entry(self):
        from ...controller.solvers import get_new_solver
        new_solver = get_new_solver(self)
        if new_solver is not None:
            if new_solver['category'] == 'filter':
                return FilterSolver(new_solver['solver'], new_solver['name'], parent=self)
            else:
                try:
                    key = new_solver['category'], new_solver['solver']
                    if key in self.local_solvers:
                        factory = self.local_solvers[key]
                    else:
                        factory = SOLVERS[key]
                except KeyError:
                    return TreeFragmentSolver.create_empty(parent=self, **new_solver)
                else:
                    return factory(new_solver['name'], parent=self)

    def stubs(self):
        return "\n".join(solver.stub() for solver in self.entries)

    def create_info(self):
        res = super(SolversModel, self).create_info()
        names = OrderedDict()
        for i, entry in enumerate(self.entries):
            if not entry.category:
                res.append(Info('Solver category is required [row: {}]'.format(i+1), Info.ERROR, rows=(i,), cols=(0,)))
            if not entry.solver and entry.category != 'filter':
                res.append(Info('Solver type is required [row: {}]'.format(i+1), Info.ERROR, rows=(i,), cols=(1,)))
            if not entry.name:
                res.append(Info('Solver name is required [row: {}]'.format(i+1), Info.ERROR, rows=(i,), cols=(2,)))
            else:
                names.setdefault(entry.name, []).append(i)
            res.extend(entry.create_info(i))
        for name, indexes in names.items():
            if len(indexes) > 1:
                res.append(Info('Duplicated solver name "{}" [rows: {}]'.format(name, ', '.join(map(str, indexes))),
                                Info.ERROR, cols=[2], rows=indexes))
        return res


def update_solvers(filename, model):
    """
    Try to load local solvers definitions
    """
    if filename is not None:
        cwd = os.path.dirname(filename)
        base, _ = os.path.splitext(filename)
        solvers_files = [os.path.join(cwd, 'solvers.yml'), base+'.yml']
    else:
        solvers_files = ['solvers.yml']
    for solvers_file in solvers_files:
        if os.path.isfile(solvers_file):
            from .schemasolver import load_yaml
            try:
                load_yaml(solvers_file, model.local_categories, model.local_solvers)
            except:
                pass
