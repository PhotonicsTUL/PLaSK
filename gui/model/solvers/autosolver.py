# coding: utf8
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

try:
    import yaml
except ImportError:
    yaml = None

from ...utils.xml import AttributeReader, print_interior
from . import Solver, SOLVERS, CATEGORIES
from .bconds import BCONDS, SchemaBoundaryConditions

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str

try:
    import plask
except ImportError:
    plask = None


class AttrList(list):
    @property
    def flat(self):
        for item in self:
            if isinstance(item, AttrGroup):
                for i in item:
                    yield i
            else:
                yield item


class AttrGroup(list):
    def __init__(self, label):
        self.label = label


class Attr(object):
    def __init__(self, tag, name, label, help, default=None):
        self.tag = tag
        self.name = name
        self.label = label
        self.help = help
        if default is None:self.default = ''
        elif isinstance(default, bool): self.default = 'yes' if default else 'no'
        else: self.default = str(default)
        self.conflicts = set()

class AttrMulti(Attr):
    pass


class AttrChoice(Attr):
    def __init__(self, tag, name, label, help, choices, default=None):
        super(AttrChoice, self).__init__(tag, name, label, help, default)
        self.choices = choices


class AttrBool(AttrChoice):
    def __init__(self, tag, name, label, help, default=None):
        super(AttrBool, self).__init__(tag, name, label, help, ('yes', 'no'), default)


class AttrGeometry(Attr):
    def __init__(self, tag, name, label, help, geometry_type, default=None):
        super(AttrGeometry, self).__init__(tag, name, label, help, default)
        self.type = geometry_type
    pass


class AttrGeometryObject(Attr):
    pass


class AttrGeometryPath(Attr):
    pass


class AttrMesh(AttrChoice):
    pass


def read_attr(tn, attr, au=None):
    an = attr['attr']
    al = attr['label']
    ah = attr['help']
    at = attr.get('type', '')
    au = attr.get('unit', au)
    ad = attr.get('default')
    if au is not None:
        al += u' [{}]'.format(au)
    if at == u'choice':
        ac = tuple(ch.strip() for ch in attr['choices'])
        result = AttrChoice(tn, an, al, ah, ac, ad)
    elif at == u'bool':
        result = AttrBool(tn, an, al, ah, ad)
    elif at == u'geometry object':
        result = AttrGeometryObject(tn, an, al, ah)
    elif at == u'geometry path':
        result = AttrGeometryPath(tn, an, al, ah)
    elif at.endswith(u' geometry'):
        result = AttrGeometry(tn, an, al, ah, at[:-9].lower())
    elif at == u'mesh':
        ac = tuple(ch.strip() for ch in attr['mesh types'])
        result = AttrMesh(tn, an, al, ah, ac)
    else:
        if at:
            ah += u' ({})'.format(at)
        if an.endswith('#'):
            result = AttrMulti(tn, an, al, ah)
        else:
            result = Attr(tn, an, al, ah, ad)
    for conflict in attr.get('conflicts', []):
        ct = conflict.get('tag', tn)
        ca = conflict['attr']
        result.conflicts.add((ct, ca))
    return result


class SchemaTag(object):
    def __init__(self, name, label, attrs):
        self.name = name
        self.label = label
        self.attrs = attrs


class AutoSolver(Solver):
    """Model for solver with its configuration specified in a scheme file solvers.xml
       and automatically generated controller widget"""

    def __init__(self, category, schema, lib=None, solver='', geometry_type=None, mesh_types=None, name='',
                 parent=None, info_cb=None):
        super(AutoSolver, self).__init__(category, solver, name, parent, info_cb)
        self.lib = lib
        self.schema = schema
        self.geometry_type = geometry_type
        self.mesh_types = mesh_types
        self.set_fresh_data()

    def set_fresh_data(self):
        self.data = dict((schema.name,
                          dict((a.name, None) for a in schema.attrs.flat) if isinstance(schema, SchemaTag) else [])
                         for schema in self.schema)

    def get_xml_element(self):
        element = etree.Element(self.category, {'name': self.name, 'solver': self.solver})
        if self.lib is not None:
            element.attrib['lib'] = self.lib
        if self.geometry:
            etree.SubElement(element, 'geometry', {'ref': self.geometry})
        if self.mesh:
            etree.SubElement(element, 'mesh', {'ref': self.mesh})
        for schema in self.schema:
            tag = schema.name
            data = self.data[tag]
            if isinstance(schema, SchemaTag):
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
            elif isinstance(schema, SchemaBoundaryConditions):
                xml = schema.to_xml(data)
                if xml is not None:
                    element.append(xml)
            else:
                if data:
                    if not isinstance(data, str): data = data.encode('utf8')
                    lines = data.split('\n')
                    if not lines[-1]: lines = lines[:-1]
                    lines = '\n    '.join(lines)
                    el = etree.fromstringlist(['<', tag, '>\n    ', lines, '\n  </', tag, '>'])
                    element.append(el)
        return element

    def set_xml_element(self, element):
        self.set_fresh_data()
        super(AutoSolver, self).set_xml_element(element)
        el = element.find('geometry')
        if el is not None:
            self.geometry = el.attrib.get('ref')
            #TODO report missing or mismatching geometry
        if self.mesh_types:
            el = element.find('mesh')
            if el is not None:
                self.mesh = el.attrib.get('ref')
                #TODO report missing or mismatching mesh
        for schema in self.schema:
            el = element.find(schema.name)
            if el is not None:
                if isinstance(schema, SchemaTag):
                    try:
                        data = self.data[schema.name]
                    except KeyError:
                        pass
                    else:
                        with AttributeReader(el) as attrs:
                            for name in attrs:
                                if name+'#' in data:
                                    data[name+'0'] = attrs[name]
                                else:
                                    data[name] = attrs[name]
                elif isinstance(schema, SchemaBoundaryConditions):
                    self.data[schema.name] = schema.from_xml(el)
                else:
                    self.data[schema.name] = print_interior(el)

    def get_controller(self, document):
        from ...controller.solvers.autosolver import AutoSolverController
        return AutoSolverController(document, self)

    def stub(self):
        if self.lib is not None:
            return "import {1}.{2}.{3} as {0}\n{0} = {0}()".format(self.name, self.category, self.lib, self.solver)
        else:
            return "import {1}.{2} as {0}\n{0} = {0}()".format(self.name, self.category, self.solver)


class AutoSolverFactory(object):

    def __init__(self, category, lib, solver, schema, geometry_type, mesh_types, providers, receivers):
        self.category = category
        self.solver = solver
        self.schema = schema
        self.geometry_type = geometry_type
        self.mesh_types = mesh_types
        self.lib = lib
        self.providers = providers
        self.receivers = receivers

    def __call__(self, name='', parent=None, info_cb=None, element=None):
        result = AutoSolver(self.category, self.schema, self.lib, self.solver, self.geometry_type, self.mesh_types,
                            name, parent, info_cb)
        if element is not None:
            result.set_xml_element(element)
        if self.lib is not None and result.lib != self.lib:
            result.lib = self.lib  # force solver lib
        return result


def _iter_tags(tags):
    for tag in tags:
        yield tag
        if 'tags' in tag:
            for t in _iter_tags(tag['tags']):
                t['tag'] = tag['tag'] + '/' + t['tag']
                yield t


def load_yaml(filename, categories=CATEGORIES, solvers=SOLVERS):
    if yaml is None: return

    for solver in yaml.load(open(filename)):
        # try:
            if not isinstance(solver, dict): continue

            name = solver.get('solver')
            if name is None: continue

            schema = []

            cat = solver.get('category')
            if cat is None: raise ValueError('Unspecified category')

            lib = solver.get('lib')
            if lib is None: lib = os.path.basename(os.path.dirname(filename)[:-4])

            geometry_type = solver.get('geometry')
            mesh_type = solver.get('mesh')
            if mesh_type:
                if isinstance(mesh_type, list):
                    mesh_types = set(mesh_type)
                    mesh_type = mesh_type[0]
                else:
                    mesh_types = {mesh_type}
            else:
                mesh_types = set()

            for tag in _iter_tags(solver['tags']):
                if 'tag' in tag:
                    tn, tl = tag['tag'], tag['label']
                    attrs = AttrList()
                    for attr in tag['attrs']:
                        if 'attr' in attr:
                            attrs.append(read_attr(tn, attr))
                        elif 'group' in attr:
                            gl = attr['group']
                            gu = attr.get('unit')
                            if gu is not None:
                                gl += u' [{}]'.format(gu)
                            group = AttrGroup(gl)
                            for a in attr.get('attrs', []):
                                group.append(read_attr(tn, a, gu))
                            attrs.append(group)
                    schema.append(SchemaTag(tn, tl, attrs))
                elif 'bcond' in tag:
                    values = tag.get('values')
                    if values is not None and not isinstance(values, list):
                        values = [values]
                    mt = tag.get('mesh type', mesh_type)
                    ma = tag.get('mesh')
                    ga = tag.get('geometry')
                    BCond = BCONDS[mt]
                    schema.append(BCond(tag['bcond'], tag['label'], tag.get("group"), mt, ma, ga, values))

            providers = [it for it in solver.get('providers', [])]
            receivers = [it for it in solver.get('receivers', [])]

            if cat not in categories:
                categories.append(cat)
            solvers[cat,name] = AutoSolverFactory(cat, lib, name, schema, geometry_type, mesh_types, providers, receivers)

        # except:
        #     continue


# Find XML files with solvers configuration
from os.path import dirname as _d

for _dirname, _, _files in os.walk(os.path.join(_d(_d(_d(_d(__file__)))), 'solvers')):
    if os.path.split(_dirname)[-1] == 'skel': continue
    for _f in _files:
        if _f.endswith('.yml'):
            load_yaml(os.path.join(_dirname, _f))
