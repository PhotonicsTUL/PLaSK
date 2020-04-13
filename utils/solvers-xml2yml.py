#!/usr/bin/python

from lxml import etree
import yaml

root = etree.parse(open('solvers.xml')).getroot()

NS = '{http://phys.p.lodz.pl/solvers.xsd}'

solvers = []

for source in root:
    solver = {}

    solver['solver'] = source.attrib['name']
    solver['lib'] = source.attrib['lib']
    solver['category'] = source.attrib['category']
    
    solver['help'] = source.text.strip()
    
    geometry = source.find(NS+'geometry')
    if geometry is not None:
        solver['geometry'] = geometry.attrib['type']
    
    mesh = source.find(NS+'mesh')
    if mesh is not None:
        solver['mesh'] = mesh.attrib['type'].replace(' ', '').split(',')
        solver['need mesh'] = not mesh.attrib.get('optional', False)
    
    tags = []
    for xmltag in source.findall(NS+'tag'):
        tag = {}
        tag['tag'] = xmltag.attrib['name']
        tag['label'] = xmltag.attrib['label']
        tag['help'] = xmltag.text.strip()
        attrs = []
        for xmlattr in xmltag.findall(NS+'attr'):
            attr = {}
            attr['attr'] = xmlattr.attrib['name']
            attr['label'] = xmlattr.attrib['label']
            attr['type'] = xmlattr.attrib['type']
            if 'unit' in xmlattr.attrib:
                attr['unit'] = xmlattr.attrib['unit']
            attr['help'] = xmlattr.text.strip()
            attrs.append(attr)
        if len(attrs) > 0: tag['attrs'] = attrs
        tags.append(tag)
    if len(tags) > 0: solver['tags'] = tags
        
    flow = source.find(NS+'flow')
    if flow is not None:
        for what in 'provider', 'receiver':
            items = []
            for item in flow.findall(NS+what):
                items.append(item.attrib['name'])
            if len(items) > 0:
                solver[what+'s'] = items

    solvers.append(solver)


with open('solvers.yml', 'w') as out:
    yaml.dump(solvers, out)
