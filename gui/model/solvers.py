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
import sys
import os
import re

from collections import OrderedDict

from ..qt import QtCore

from lxml import etree
from xml.sax.saxutils import quoteattr

from ..utils.xml import print_interior, XML_parser, AttributeReader
from ..controller.source import SourceEditController
from ..controller.solvers import ConfSolverController, FilterController
from .table import TableModel
from . import TreeFragmentModel, Info

SOLVERS = {}

CATEGORIES = (
    'thermal',
    'electrical',
    'gain',
    'optical'
)


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


class ConfSolver(Solver):
    """Model for solver with its configuration specified in a simple Python dictionary
       and automatically generated controller widget"""

    def __init__(self, category, config, lib=None, solver='', mesh_type=None, name='', parent=None, info_cb=None):
        super(ConfSolver, self).__init__(category, solver, name, parent, info_cb)
        self.lib = lib
        self.config = config
        self.mesh_type = mesh_type
        self.set_fresh_data()

    def set_fresh_data(self):
        self.data = dict((tag,
                          dict((a[0], '') for a in attrs) if type(attrs) in (tuple, list) else
                          ''  # TODO add proper support for boundary conditions
                         ) for (tag, _, attrs) in self.config)

    def get_xml_element(self):
        element = etree.Element(self.category, {'name': self.name, 'solver': self.solver})
        if self.lib is not None:
            element.attrib['lib'] = self.lib
        if self.geometry:
            etree.SubElement(element, 'geometry', {'ref': self.geometry})
        if self.mesh:
            etree.SubElement(element, 'mesh', {'ref': self.mesh})
        for tag,_,_ in self.config:
            data = self.data[tag]
            if type(data) is dict:
                attrs = dict((item for item in data.items() if item[1] and item[0][-1] != '#'))
                if attrs:
                    if '/' in tag:
                        path = tag.split('/')
                        tag = path[-1]
                        el = element
                        for tg in path[:-1]:
                            f = el.find(tg)
                            if f is not None:
                                el = f
                            else:
                                el = etree.SubElement(el, tg)
                        etree.SubElement(el, tag, attrs)
                    else:
                        etree.SubElement(element, tag, attrs)
            else:
                if data:
                    lines = data.encode('utf-8').split('\n')
                    if not lines[-1]: lines = lines[:-1]
                    lines = '\n    '.join(lines)
                    el = etree.fromstringlist(['<', tag, '>\n    ', lines, '\n  </', tag, '>'])
                    element.append(el)
        return element

    def set_xml_element(self, element):
        self.set_fresh_data()
        super(ConfSolver, self).set_xml_element(element)
        el = element.find('geometry')
        if el is not None:
            self.geometry = el.attrib.get('ref')
            #TODO report missing or mismatching geometry
        if self.mesh_type:
            el = element.find('mesh')
            if el is not None:
                self.mesh = el.attrib.get('ref')
                #TODO report missing or mismatching mesh
        for tag,_,_ in self.config:
            el = element.find(tag)
            if el is not None:
                try:
                    data = self.data[tag]
                except KeyError:
                    pass
                else:
                    if type(data) is dict:
                        with AttributeReader(el) as attrs:
                            for name in attrs:
                                if name+'#' in data:
                                    data[name+'0'] = attrs[name]
                                else:
                                    data[name] = attrs[name]
                    else:
                        self.data[tag] = print_interior(el)

    def get_controller(self, document):
        return ConfSolverController(document, self)

    def stub(self):
        if self.lib is not None:
            return "import {1}.{2}.{3} as {0}\n{0} = {0}()".format(self.name, self.category, self.lib, self.solver)
        else:
            return "import {1}.{2} as {0}\n{0} = {0}()".format(self.name, self.category, self.solver)


class ConfSolverFactory(object):

    def __init__(self, category, lib, solver, config, mesh_type, providers, receivers):
        self.category = category
        self.solver = solver
        self.config = config
        self.mesh_type = mesh_type
        self.lib = lib
        self.providers = providers
        self.receivers = receivers

    def __call__(self, name='', parent=None, info_cb=None, element=None):
        result = ConfSolver(self.category, self.config, self.lib, self.solver, self.mesh_type, name, parent, info_cb)
        if element is not None:
            result.set_xml_element(element)
        return result


class FilterSolver(Solver):

    def __init__(self, what='', name='', parent=None, info_cb=None):
        super(FilterSolver, self).__init__('filter', parent=parent, info_cb=info_cb)
        self.what = what
        self.name = name
        self.geometry = ''

    def get_xml_element(self):
        return etree.Element(self.category, {"name": self.name, "for": self.what, "geometry": self.geometry})

    def set_xml_element(self, element):
        self.category = element.tag
        with AttributeReader(element) as attr:
            self.name = attr.get('name', None)
            self.what = attr.get('for', None)
            self.geometry = attr.get('geometry', None)

    def get_controller(self, document):
        return FilterController(document, self)

    def stub(self):
        return "{} = flow.{}Filter()".format(self.name, self.what)  # TODO: Geometry suffix


class SolversModel(TableModel):

    def __init__(self, parent=None, info_cb=None, *args):
        super(SolversModel, self).__init__('solvers', parent, info_cb, *args)

    def construct_solver(self, element):
        if element.tag == 'filter':
            filter = FilterSolver(parent=self)
            filter.set_xml_element(element)
            return filter
        else:
            try:
                factory = SOLVERS[element.tag, element.attrib['solver']]
            except KeyError:
                return TreeFragmentSolver(element, self)
            else:
                return factory(element=element, parent=self)

    def set_xml_element(self, element, undoable=True):
        self._set_entries([] if element is None else [self.construct_solver(item) for item in element], undoable)

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
        if index.column() < 2: flags &= ~QtCore.Qt.ItemIsEditable
        return flags

    def create_default_entry(self):
        from ..controller.solvers import get_new_solver
        new_solver = get_new_solver()
        if new_solver is not None:
            if new_solver['category'] == 'filter':
                return FilterSolver(new_solver['solver'], new_solver['name'], parent=self)
            else:
                try:
                    factory = SOLVERS[new_solver['category'], new_solver['solver']]
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
        for name, indexes in names.items():
            if len(indexes) > 1:
                res.append(Info('Duplicated solver name "{}" [rows: {}]'.format(name, ', '.join(map(str, indexes))),
                                Info.ERROR, cols=[2], rows=indexes))
        return res


# Find XML files with solvers configuration
from os.path import dirname as _d

XNS = '{http://phys.p.lodz.pl/solvers.xsd}'


def _iter_tags(parent):
    for tag in parent.findall(XNS+'tag'):
        yield tag
        for t in _iter_tags(tag):
            t.attrib['name'] = tag.attrib['name'] + '/' + t.attrib['name']
            yield t


def _load_xml(filename):

    cat = os.path.basename(_d(filename))
    if cat == 'skel': return

    dom = etree.parse(filename)
    root = dom.getroot()

    if root.tag != XNS+'solvers': return

    for solver in root:
        if solver.tag != XNS+'solver': return

        name = solver.attrib.get('name')
        if name is None: return

        cat = solver.attrib.get('category', cat)
        lib = solver.attrib.get('lib', os.path.basename(filename)[:-4])

        m = solver.find(XNS+'mesh')
        if m is not None:
            try:
                mesh_type = m.attrib['type']
            except KeyError:
                mesh_type = None
        else:
            mesh_type = None

        config = []

        for tag in _iter_tags(solver):
            tn, tl = tag.attrib['name'], tag.attrib['label']
            attrs = []
            for attr in tag.findall(XNS+'attr'):
                an = attr.attrib['name']
                al = attr.attrib['label']
                ah = attr.text
                at = attr.attrib.get('type', '')
                au = attr.attrib.get('unit', None)
                if au is not None:
                    al += u' [{}]'.format(au)
                    at += u' [{}]'.format(au)
                if at == u'choice':
                    ac = tuple(ch.text.strip() for ch in attr.findall(XNS+'choice'))
                    attrs.append((an, al, ah, ac))
                elif at == u'bool':
                    attrs.append((an, al, ah, ('yes', 'no')))
                else:
                    if at:
                        ah += u' ({})'.format(at)
                    attrs.append((an, al, ah))
            config.append((tn, tl, attrs))

        #TODO Handle boundary conditions properly
        for bcond in solver.findall(XNS+'bcond'):
            config.append((bcond.attrib['name'], bcond.attrib['label'] + ' boundary conditions', None))

        flow = solver.find(XNS+'flow')
        if flow is not None:
            providers = [(e.attrib['name'], e.attrib.get('for', e.attrib['name'][3:]))
                         for e in flow.findall(XNS+'provider')]
            receivers = [(e.attrib['name'], e.attrib.get('for', e.attrib['name'][2:]))
                         for e in flow.findall(XNS+'receiver')]
        else:
            providers = []
            receivers = []

        SOLVERS[cat,name] = ConfSolverFactory(cat, lib, name, config, mesh_type, providers, receivers)

for _dirname, _, _files in os.walk(os.path.join(_d(_d(_d(__file__))), 'solvers')):
    for _f in _files:
        if _f.endswith('.xml'):
            _load_xml(os.path.join(_dirname, _f))
