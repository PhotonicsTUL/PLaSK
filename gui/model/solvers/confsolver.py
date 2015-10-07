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
import os
from lxml import etree

from ...utils.xml import AttributeReader, print_interior
from . import Solver, SOLVERS


class Attr(object):
    def __init__(self, name, label, help):
        self.name = name
        self.label = label
        self.help = help


class AttrMulti(Attr):
    pass


class AttrChoice(Attr):
    def __init__(self, name, label, help, choices):
        super(AttrChoice, self).__init__(name, label, help)
        self.choices = choices


class AttrBool(AttrChoice):
    def __init__(self, name, label, help):
        super(AttrBool, self).__init__(name, label, help, ('yes', 'no'))


class AttrGeometryObject(Attr):
    pass


class AttrGeometryPath(Attr):
    pass


def read_attr(attr, xns):
        an = attr.attrib['name']
        al = attr.attrib['label']
        ah = attr.text
        at = attr.attrib.get('type', '')
        au = attr.attrib.get('unit', None)
        if au is not None:
            al += u' [{}]'.format(au)
            at += u' [{}]'.format(au)
        if at == u'choice':
            ac = tuple(ch.text.strip() for ch in attr.findall(xns+'choice'))
            return AttrChoice(an, al, ah, ac)
        elif at == u'bool':
            return AttrBool(an, al, ah)
        elif at == u'geometry object':
            return AttrGeometryObject(an, al, ah)
        elif at == u'geometry path':
            return AttrGeometryPath(an, al, ah)
        else:
            if at:
                ah += u' ({})'.format(at)
            if an.endswith('#'):
                return AttrMulti(an, al, ah)
            else:
                return Attr(an, al, ah)


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
                          dict((a.name, '') for a in attrs) if type(attrs) in (tuple, list) else
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
        from ...controller.solvers import ConfSolverController
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
        if self.lib is not None and result.lib != self.lib:
            result.lib = self.lib  # force solver lib
        return result


def _iter_tags(parent, xns):
    for tag in parent.findall(xns+'tag'):
        yield tag
        for t in _iter_tags(tag, xns):
            t.attrib['name'] = tag.attrib['name'] + '/' + t.attrib['name']
            yield t


def _load_xml(filename):

    cat = os.path.basename(_d(filename))
    if cat == 'skel': return

    dom = etree.parse(filename)
    root = dom.getroot()

    xns = root.nsmap.get(None, '')
    if xns: xns = '{'+xns+'}'

    if root.tag != xns+'solvers': return

    for solver in root:
        if solver.tag != xns+'solver': return

        name = solver.attrib.get('name')
        if name is None: return

        cat = solver.attrib.get('category', cat)
        lib = solver.attrib.get('lib', os.path.basename(filename)[:-4])

        m = solver.find(xns+'mesh')
        if m is not None:
            try:
                mesh_type = m.attrib['type']
            except KeyError:
                mesh_type = None
        else:
            mesh_type = None

        config = []

        for tag in _iter_tags(solver, xns):
            tn, tl = tag.attrib['name'], tag.attrib['label']
            attrs = []
            for attr in tag.findall(xns+'attr'):
                attrs.append(read_attr(attr, xns))
            config.append((tn, tl, attrs))

        #TODO Handle boundary conditions properly
        for bcond in solver.findall(xns+'bcond'):
            config.append((bcond.attrib['name'], bcond.attrib['label'] + ' boundary conditions', None))

        flow = solver.find(xns+'flow')
        if flow is not None:
            providers = [(e.attrib['name'], e.attrib.get('for', e.attrib['name'][3:]))
                         for e in flow.findall(xns+'provider')]
            receivers = [(e.attrib['name'], e.attrib.get('for', e.attrib['name'][2:]))
                         for e in flow.findall(xns+'receiver')]
        else:
            providers = []
            receivers = []

        SOLVERS[cat,name] = ConfSolverFactory(cat, lib, name, config, mesh_type, providers, receivers)


# Find XML files with solvers configuration
from os.path import dirname as _d

for _dirname, _, _files in os.walk(os.path.join(_d(_d(_d(_d(__file__)))), 'solvers')):
    for _f in _files:
        if _f.endswith('.xml'):
            _load_xml(os.path.join(_dirname, _f))
