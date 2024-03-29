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

# coding: utf8

import sys
import os
from importlib import import_module

from lxml import etree

import yaml

from ..info import Info
from ...utils.xml import AttributeReader, print_interior, UnorderedTagReader
from ...utils.validators import can_be_int, can_be_float, can_be_one_of, can_be_bool
from ...utils.files import open_utf8
from . import Solver, Tag, SOLVERS, CATEGORIES
from .bconds import BCONDS, SchemaBoundaryConditions

from ...lib import yaml_include

# Disable yaml warning
try:
    yaml.warnings({'YAMLLoadWarning': False})
except (TypeError, NameError, AttributeError):
    pass

VALIDATORS = {'int': can_be_int, 'float': can_be_float, 'bool': can_be_bool}

basestring = str, bytes

try:
    import plask
except ImportError:
    plask = None
    _prefix = sys.prefix
else:
    _prefix = plask.prefix

yaml_include.AddYamlIncludePath(os.path.join(_prefix, 'share', 'plask', 'schemas'))


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


class Attr:
    def __init__(self, tag, name, label, required, help, typ, default=None):
        self.tag = tag
        self.name = name
        self.label = label
        self.required = required
        self.help = help
        self.typ = typ
        if default is None: self.default = ''
        elif isinstance(default, bool): self.default = 'yes' if default else 'no'
        else: self.default = str(default)
        self.conflicts = set()

class AttrMulti(Attr):
    pass


class AttrChoice(Attr):
    def __init__(self, tag, name, label, required, help, typ, choices, other=(), cs=False, default=None):
        super().__init__(tag, name, label, required, help, typ, default)
        self.choices = choices
        self.other = other
        self.cs = cs


class AttrBool(AttrChoice):
    def __init__(self, tag, name, label, required, help, typ, default=None):
        super().__init__(tag, name, label, required, help, typ,
                                       ('yes', 'no'), ('true', 'false', '1', '0'), default=default)


class AttrGeometry(Attr):
    def __init__(self, tag, name, label, required, help, typ, geometry_type, default=None):
        super().__init__(tag, name, label, required, help, typ, default)
        self.type = geometry_type


class AttrGeometryObject(Attr):
    pass


class AttrGeometryPath(Attr):
    pass


class AttrMesh(Attr):
    def __init__(self, tag, name, label, required, help, typ, mesh_types, default=None):
        super().__init__(tag, name, label, required, help, typ, default)
        self.types = mesh_types


class AttrMaterial(Attr):
    def __init__(self, tag, name, label, required, help, typ, default=None):
        super().__init__(tag, name, label, required, help, typ, default)


class SchemaTag:
    def __init__(self, name, label, attrs):
        self.name = name
        self.label = label
        self.attrs = attrs


class SchemaGenericTag:
    def __init__(self, name, label):
        self.name = name
        self.label = label


class SchemaCustomWidgetTag:
    def __init__(self, name, label, button_label, edit_func):
        self.name = name
        self.label = label
        self.button_label = button_label
        self.edit_func = edit_func


class SchemaSolver(Solver):
    """Model for solver with its configuration specified in a scheme file solvers.yml
       and automatically generated controller widget"""

    def __init__(self, category, schema, lib=None, solver='', geometry_type=None, mesh_types=None,
                 need_mesh=True, name='', parent=None, info_cb=None):
        super().__init__(category, solver, name, parent, info_cb)
        self.lib = lib
        self.schema = schema
        self.geometry_type = geometry_type
        self.mesh_types = mesh_types
        self.need_mesh = need_mesh
        self.set_fresh_data()
        self.incomments = {}
        self._controller = None

    def set_fresh_data(self):
        self.data = {schema.name:
                         {a.name: None for a in schema.attrs.flat} if isinstance(schema, SchemaTag) else
                         Tag(schema.name) if isinstance(schema, SchemaCustomWidgetTag) else
                         '' if isinstance(schema, SchemaGenericTag) else []
                     for schema in self.schema}

    def make_xml_element(self):
        element = etree.Element(self.category, {'name': self.name, 'solver': self.solver})
        if self.lib is not None:
            element.attrib['lib'] = self.lib
        if self.geometry:
            for c in self.incomments.get('geometry', ()):
                element.append(etree.Comment(c))
            etree.SubElement(element, 'geometry', {'ref': self.geometry})
        if self.mesh:
            for c in self.incomments.get('mesh', ()):
                element.append(etree.Comment(c))
            etree.SubElement(element, 'mesh', {'ref': self.mesh})
        done = {}
        for schema in self.schema:
            tag = schema.name
            data = self.data[tag]
            if isinstance(schema, SchemaTag):
                attrs = dict((item for item in data.items() if item[1] and item[0][-1] != '#'))
                if '/' in tag:
                    path = tag.split('/')
                    tag = path[-1]
                    el = element
                    ck = None
                    for tg in path[:-1]:
                        ck = tg if ck is None else ck+'/'+tg
                        f = el.find(tg)
                        if f is not None:
                            el = f
                        else:
                            for c in self.incomments.get(ck, ()):
                                el.append(etree.Comment(c))
                            el = etree.SubElement(el, tg)
                            done[ck] = el
                    ck += '/'+tag
                    for c in self.incomments.get(ck, ()):
                        el.append(etree.Comment(c))
                    done[ck] = etree.SubElement(el, tag, attrs)
                else:
                    el = element.find(tag)
                    if el is None:
                        for c in self.incomments.get(tag, ()):
                            element.append(etree.Comment(c))
                        done[tag] = etree.SubElement(element, tag, attrs)
                    else:
                        el.attrib.update(attrs)
            elif isinstance(schema, SchemaCustomWidgetTag):
                if data:
                    for c in self.incomments.get(schema.name, ()):
                        element.append(etree.Comment(c))
                    element.append(data.make_xml_element())
            elif isinstance(schema, SchemaBoundaryConditions):
                for c in self.incomments.get(schema.name, ()):
                    element.append(etree.Comment(c))
                xml = schema.to_xml(data)
                if xml is not None:
                    element.append(xml)
            else:
                for c in self.incomments.get(tag, ()):
                    element.append(etree.Comment(c))
                if data:
                    lines = data.split('\n')
                    if not lines[-1]: lines = lines[:-1]
                    lines = '\n    '.join(lines)
                    el = etree.fromstringlist(['<', tag, '>\n    ', lines, '\n  </', tag, '>'])
                    element.append(el)
                    done[tag] = el
        for ck, el in reversed(done.items()):
            for c in self.incomments.get(ck+'/', ()):
                el.append(etree.Comment(c))
            if len(el) == 0 and len(el.attrib) == 0 and el.text is None:
                el.getparent().remove(el)
        for c in self.incomments.get('/', ()):
            element.append(etree.Comment(c))
        return element

    def load_xml_element(self, element):
        self.set_fresh_data()
        super().load_xml_element(element)
        self.incomments = {}
        with UnorderedTagReader(element) as reader:
            el = reader.find('geometry')
            if el is not None:
                self.geometry = el.attrib.get('ref')
                #TODO report missing or mismatching geometry
                self.incomments['geometry'] = el.comments
            if self.mesh_types:
                el = reader.find('mesh')
                if el is not None:
                    self.mesh = el.attrib.get('ref')
                    #TODO report missing or mismatching mesh
                    self.incomments['mesh'] = el.comments
            groups = set()
            for schema in self.schema:
                el = reader.find(schema.name)
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
                    elif isinstance(schema, SchemaCustomWidgetTag):
                        self.data[schema.name] = Tag.from_xml(el)
                    elif isinstance(schema, SchemaBoundaryConditions):
                        self.data[schema.name] = schema.from_xml(el)
                    else:
                        self.data[schema.name] = print_interior(el)
                    self.incomments[schema.name] = el.comments
                    if '/' in schema.name:
                        groups.add('/'.join(schema.name.split('/')[:-1]))
            for group in groups:
                el = reader.find(group)
                self.incomments[group] = el.comments
                self.incomments[group+'/'] = reader.get_comments(el, True)
            self.incomments['/'] = reader.get_comments()

    def get_controller(self, document):
        from ...controller.solvers.schemasolver import SchemaSolverController
        if self._controller is None:
            self._controller = SchemaSolverController(document, self)
        return self._controller

    def _create_info_for_attrs(self, res, row, tag, attrs):
        for attr in attrs:
            if isinstance(attr, AttrGroup):
                self._create_info_for_attrs(res, row, tag, attr)
            elif isinstance(attr, Attr):
                value = self.data[tag.name][attr.name]
                if attr.required and not value:
                    res.append(Info("{attr.label} (in {tag.label}) required in solver '{name}' [row: {row}]"
                                    .format(attr=attr, tag=tag, name=self.name, row=row+1),
                                    Info.ERROR, rows=(row,), what=(tag.name, attr.name)))
                elif attr.typ in VALIDATORS and not VALIDATORS[attr.typ](value):
                    res.append(Info("{typ} value required for {attr.label} (in {tag.label})"
                                    " in solver '{name}' [row: {row}]"
                                    .format(attr=attr, tag=tag, name=self.name, row=row+1, typ=attr.typ.title()),
                                    Info.ERROR, rows=(row,), what=(tag.name, attr.name)))
                elif isinstance(attr, AttrChoice) and not can_be_one_of(value, *(attr.choices + attr.other),
                                                                        case_sensitive=attr.cs):
                    res.append(Info("{attr.label} (in {tag.label}) must be one of {choices}"
                                    " in solver '{name}' [row: {row}]"
                                    .format(attr=attr, tag=tag, name=self.name, row=row+1,
                                            choices=', '.join("'{}'".format(c) for c in (attr.choices + attr.other))),
                                    Info.ERROR, rows=(row,), what=(tag.name, attr.name)))

    def create_info(self, row):
        res = super().create_info(row)
        if not self.geometry and self.geometry_type is not None:
            res.append(Info("Geometry required for solver '{}' [row: {}]".format(self.name, row+1),
                            Info.ERROR, rows=(row,), what='geometry'))
        if not self.mesh and self.mesh_types and self.need_mesh:
            res.append(Info("Mesh required for solver '{}' [row: {}]".format(self.name, row+1),
                            Info.ERROR, rows=(row,), what='mesh'))
        for tag in self.schema:
            if isinstance(tag, SchemaTag):
                self._create_info_for_attrs(res, row, tag, tag.attrs)
        return res

    def stub(self):
        if self.lib is not None:
            return "from {1}.{2} import {3} as {0}\n{0} = {0}()".format(self.name, self.category, self.lib, self.solver)
        else:
            return "from {1} import {2} as {0}\n{0} = {0}()".format(self.name, self.category, self.solver)


class SchemaSolverFactory:

    def __init__(self, category, lib, solver, schema, geometry_type, mesh_types, need_mesh, providers, receivers):
        self.category = category
        self.solver = solver
        self.schema = schema
        self.geometry_type = geometry_type
        self.mesh_types = mesh_types
        self.need_mesh = need_mesh
        self.lib = lib
        self.providers = providers
        self.receivers = receivers

    def __call__(self, name='', parent=None, info_cb=None, element=None):
        result = SchemaSolver(self.category, self.schema, self.lib, self.solver, self.geometry_type, self.mesh_types,
                              self.need_mesh, name, parent, info_cb)
        if element is not None:
            result.load_xml_element(element)
        if self.lib is not None and result.lib != self.lib:
            result.lib = self.lib  # force solver lib
        return result


def read_attr(tn, attr):
    an = attr['attr']
    al = attr['label']
    ah = attr['help'].strip()
    at = attr.get('type', '')
    au = attr.get('unit')
    ad = attr.get('default')
    ar = attr.get('required', False)
    if au is not None:
        al += ' ({})'.format(au)
    if at == 'choice':
        ac = tuple(str(ch).strip() for ch in attr['choices'])
        ao = tuple(str(ch).strip() for ch in attr.get('other', ()))
        ak = attr.get('case sensitive', False)
        result = AttrChoice(tn, an, al, ar, ah, at, ac, ao, ak, ad)
    elif at == 'bool':
        result = AttrBool(tn, an, al, ar, ah, at, ad)
    elif at == 'geometry object':
        result = AttrGeometryObject(tn, an, al, ar, ah, at)
    elif at == 'geometry path':
        result = AttrGeometryPath(tn, an, al, ar, ah, at)
    elif at.endswith(' geometry'):
        result = AttrGeometry(tn, an, al, ar, ah, at, at[:-9].lower())
    elif at == 'mesh':
        ac = tuple(str(ch).strip() for ch in attr['mesh types'])
        result = AttrMesh(tn, an, al, ar, ah, at, ac)
    elif at == 'material':
        result = AttrMaterial(tn, an, al, ar, ah, at)
    else:
        if an.endswith('#'):
            result = AttrMulti(tn, an, al, ar, ah, at)
        else:
            result = Attr(tn, an, al, ar, ah, at, ad)
    for conflict in attr.get('conflicts', []):
        ct = conflict.get('tag', tn)
        ca = conflict['attr']
        result.conflicts.add((ct, ca))
    return result


def _iter_tags(tags):
    for tag in tags:
        yield tag
        if 'tags' in tag:
            if 'widget' in tag:
                raise TypeError("Solver schema cannot specify both 'tags' and 'widget'")
            for t in _iter_tags(tag['tags']):
                t = t.copy()
                t['tag'] = tag['tag'] + '/' + t['tag']
                yield t


def load_yaml(filename, categories=CATEGORIES, solvers=SOLVERS):
    if yaml is None: return

    for solver in yaml.safe_load(open_utf8(filename)):
        try:
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
            need_mesh = solver.get('need mesh', True)
            if mesh_type:
                if isinstance(mesh_type, list):
                    mesh_types = set(mesh_type)
                    mesh_type = mesh_type[0]
                else:
                    mesh_types = {mesh_type}
            else:
                mesh_types = set()

            for tag in _iter_tags(solver.get('tags', [])):
                if 'tag' in tag:
                    tn, tl = tag['tag'], tag['label']
                    if 'widget' in tag:
                        if 'attrs' in tag:
                            raise TypeError("Solver schema cannot specify both 'attrs' and 'widget'")
                        widget = tag['widget']
                        if widget is None:
                            schema.append(SchemaGenericTag(tn, tl))
                        else:
                            modname = widget['file']
                            if modname.endswith('__init__.py'):
                                modname = modname[:-11]
                            elif modname.endswith('.py'):
                                modname = modname[:-3]
                            else:
                                raise ValueError("Widget file '{}' in '{}' tag is not a Python file".format(modname, tn))
                            modpath, modname = os.path.split(modname)
                            if not os.path.isabs(modpath):
                                modpath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(filename)), modpath))
                            sys_path = sys.path[:]
                            sys.path.insert(0, modpath)
                            try:
                                mod = import_module(modname)
                            finally:
                                sys.path[:] = sys_path
                            func = mod.__dict__[widget['func']]
                            schema.append(SchemaCustomWidgetTag(tn, tl, widget.get('button label', "View / Edit"), func))
                    else:
                        attrs = AttrList()
                        for attr in tag.get('attrs', []):
                            if 'attr' in attr:
                                attrs.append(read_attr(tn, attr))
                            elif 'group' in attr:
                                gl = attr['group']
                                gu = attr.get('unit')
                                if gu is not None:
                                    gl += ' ({})'.format(gu)
                                group = AttrGroup(gl)
                                for a in attr.get('attrs', []):
                                    group.append(read_attr(tn, a))
                                attrs.append(group)
                        schema.append(SchemaTag(tn, tl, attrs))
                elif 'bcond' in tag:
                    values = tag.get('values')
                    if values is not None and not isinstance(values, list):
                        values = [values]
                    mt = tag.get('mesh type', mesh_type)
                    if mt is not None:
                        ma = tag.get('mesh')
                        ga = tag.get('geometry')
                        BCond = BCONDS[mt]
                        schema.append(BCond(tag['bcond'], tag['label'], tag.get("group"), mt, ma, ga, values))

            providers = [tuple(it.items())[0] if isinstance(it, dict) else (it, it[3:])
                         for it in solver.get('providers', [])]
            receivers = [tuple(it.items())[0] if isinstance(it, dict) else (it, it[2:])
                         for it in solver.get('receivers', [])]

            if cat not in categories:
                categories.append(cat)
            solvers[cat,name] = SchemaSolverFactory(cat, lib, name, schema, geometry_type, mesh_types, need_mesh,
                                                    providers, receivers)
        except:
            from ... import _DEBUG
            if _DEBUG:
                import traceback
                traceback.print_exc()


# Find XML files with solvers configuration
from os.path import dirname as _d

for _dirname, _, _files in os.walk(os.path.join(_d(_d(_d(_d(__file__)))), 'solvers')):
    if os.path.split(_dirname)[-1] == 'skel': continue
    for _f in _files:
        if _f.endswith('.yml'):
            try:
                load_yaml(os.path.join(_dirname, _f))
            except yaml.YAMLError:
                from ... import _DEBUG
                if _DEBUG:
                    import traceback
                    traceback.print_exc()
