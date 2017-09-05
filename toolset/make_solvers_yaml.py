#!/usr/bin/python
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

import sys
import os.path
from lxml import etree
import yaml
from collections import OrderedDict


def represent_odict(dump, tag, mapping, flow_style=None):
    value = []
    node = yaml.MappingNode(tag, value, flow_style=flow_style)
    if dump.alias_key is not None:
        dump.represented_objects[dump.alias_key] = node
    best_style = True
    if hasattr(mapping, 'items'):
        mapping = mapping.items()
    for item_key, item_value in mapping:
        node_key = dump.represent_data(item_key)
        node_value = dump.represent_data(item_value)
        if not (isinstance(node_key, yaml.ScalarNode) and not node_key.style):
            best_style = False
        if not (isinstance(node_value, yaml.ScalarNode) and not node_value.style):
            best_style = False
        value.append((node_key, node_value))
    if flow_style is None:
        if dump.default_flow_style is not None:
            node.flow_style = dump.default_flow_style
        else:
            node.flow_style = best_style
    return node

yaml.SafeDumper.add_representer(OrderedDict,
                                lambda dumper, value: represent_odict(dumper, u'tag:yaml.org,2002:map', value))


class AnchorSafeDumper(yaml.SafeDumper):
    _suffixes = {}
    def generate_anchor(self, node):
        try:
            # label = '-'.join(n.value for n in node.value[0])
            label = node.value[0][1].value
        except TypeError:
            if node.value[0].value.startswith('out'):
                label = 'providers'
            elif node.value[0].value.startswith('in'):
                label = 'receivers'
        if label in self._suffixes:
            i = self._suffixes[label] = self._suffixes[label] + 1
            label += str(i)
        else:
            self._suffixes[label] = 1
        print("Generating anchor: {}".format(label))
        return label


def optional(source, xml_name, target, yaml_name=None):
    if yaml_name is None: yaml_name = xml_name
    value = source.attrib.get(xml_name)
    if value is not None: target[yaml_name] = value


def parse_text(txt):
    return u' '.join(l.strip() for l in txt.splitlines()).strip()


def parse(val):
    try:
        val = eval(val)
    except:
        pass
    else:
        if type(val) == complex: val = str(val)
    return val


ATTRS = []


def read_attr(attr, xns):
    info = OrderedDict()
    info['attr'] = attr.attrib['name']
    info['label'] = attr.attrib['label']
    if attr.attrib.get('required') == 'yes':
        info['required'] = True
    at = attr.attrib.get('type')
    if at is not None:
        info['type'] = at
    optional(attr, 'unit', info)
    ad = attr.attrib.get('default')
    if ad is not None:
        if at == 'bool':
            info['default'] = ad.lower() == 'yes'
        else:
            info['default'] = parse(ad)
    if at == u'choice':
        info['choices'] = [ch.text.strip() for ch in attr.findall(xns + 'choice')]
    elif at == u'mesh':
        info['mesh types'] = [ch.text.strip() for ch in attr.findall(xns + 'type')]
    conflicts = []
    for conflict in attr.findall(xns + 'conflicts'):
        ct = conflict.attrib.get('tag')
        ci = OrderedDict()
        if ct is not None:
            ci['tag'] = ct
        ci['attr'] = conflict.attrib['attr']
        conflicts.append(ci)
    if conflicts:
        info['conflicts'] = conflicts
    info['help'] = parse_text(attr.text)
    for existing in ATTRS:
        if info == existing:
            info = existing
            break
    else:
        ATTRS.append(info)
    return info


TAGS = []


def iter_tags(parent, ns, xns, target):
    for tag in parent.xpath('p:tag|p:bcond', namespaces={'p': ns}):
        info = OrderedDict()
        if tag.tag == xns + 'tag':
            info['tag'] = tag.attrib['name']
            info['label'] = tag.attrib['label']
            info['help'] = parse_text(tag.text)
            info['attrs'] = attrs = []
            for attr in tag.iterchildren(xns + 'attr', xns + 'group'):
                if attr.tag == xns + 'attr':
                    attrs.append(read_attr(attr, xns))
                elif attr.tag == xns + 'group':
                    gi = OrderedDict()
                    gi['group'] = attr.attrib['label']
                    optional(attr, 'unit', gi)
                    gi['attrs'] = []
                    for a in attr.findall(xns + 'attr'):
                        gi['attrs'].append(read_attr(a, xns))
                    attrs.append(gi)
            subtags = []
            iter_tags(tag, ns, xns, subtags)
            if subtags:
                info['tags'] = subtags
        elif tag.tag == xns + 'bcond':
            info['bcond'] = tag.attrib['name']
            info['label'] = tag.attrib['label']
            values = tag.attrib.get('values')
            if values is not None:
                values = values.split(',')
                info['values'] = values
            optional(tag, 'type', info, 'mesh type')
            mesh = tag.attrib.get('mesh')
            if mesh:
                mi = info['mesh'] = OrderedDict()
                mi['tag'], mi['attr'] = mesh.split(':')
            geometry = tag.attrib.get('geometry')
            if mesh:
                gi = info['geometry'] = OrderedDict()
                gi['tag'], gi['attr'] = geometry.split(':')
            if tag.text:
                info['help'] = parse_text(tag.text)
        for existing in TAGS:
            if info == existing:
                info = existing
                break
        else:
            TAGS.append(info)
        target.append(info)


FLOW = []


def proceed(filename):
    dom = etree.parse(filename)
    root = dom.getroot()

    ns = root.nsmap.get(None, '')
    xns = '{'+ns+'}' if ns else ''

    if root.tag != xns+'solvers': return

    data = []
    templates = {}

    for solver in root:
        if solver.tag not in (xns+'solver', xns+'template'):
            continue
        if solver.attrib.get('obsolete'):
            continue

        info = OrderedDict()
        info['solver'] = solver.attrib['name']

        template = templates.get(solver.attrib.get('template'))

        if template is not None and 'lib' in template:
            info['lib'] = template['lib']
        optional(solver, 'lib', info)

        if template is not None and 'category' in template:
            info['category'] = template['category']
        optional(solver, 'category', info)

        if template is not None and 'geometry' in template:
            info['geometry'] = template['geometry']
        geom = solver.find(xns+'geometry')
        if geom is not None:
            info['geometry'] = geom.attrib['type']

        if template is not None and 'mesh' in template:
            info['mesh'] = template['mesh']
        mesh = solver.find(xns+'mesh')
        if mesh is not None:
            try:
                mesh_types = [mesh.attrib['type']]
            except KeyError:
                mesh_types = []
            for t in mesh.findall(xns+'type'):
                mesh_types.append(t.text)
            if len(mesh_types) == 1:
                info['mesh'] = mesh_types[0]
            else:
                info['mesh'] = mesh_types

        info['tags'] = tags = []
        iter_tags(solver, ns, xns, tags)

        if template is not None and 'tags' in template:
            tags.extend(template['tags'])

        flow = solver.find(xns+'flow')
        if flow is not None:
            providers = [e.attrib['name'] if e.attrib.get('for') is None else
                         {e.attrib['name']: e.attrib['for']} for e in flow.findall(xns+'provider')]
            receivers = [e.attrib['name'] if e.attrib.get('for') is None else
                         {e.attrib['name']: e.attrib['for']} for e in flow.findall(xns+'receiver')]
        else:
            providers = []
            receivers = []
        if template is not None and 'providers' in template:
            providers.extend(template['providers'])
        if template is not None and 'receivers' in template:
            receivers.extend(template['receivers'])
        for existing in FLOW:
            if providers == existing:
                providers = existing
                break
        else:
            FLOW.append(providers)
        for existing in FLOW:
            if receivers == existing:
                receivers = existing
                break
        else:
            FLOW.append(receivers)
        if providers: info['providers'] = providers
        if receivers: info['receivers'] = receivers

        if solver.tag == xns + 'solver':
            data.append(info)
        elif solver.tag == xns+'template':
            templates[solver.attrib['name']] = info

    outname = os.path.join(os.path.dirname(os.path.abspath(filename)), "solvers.yml")
    print("Writing {}".format(outname))
    with open(outname, 'w') as out:
        yaml.dump(data, out, encoding='utf-8', allow_unicode=True, width=102, default_flow_style=False,
                  Dumper=AnchorSafeDumper)

if __name__ == "__main__":
    proceed(sys.argv[1])
