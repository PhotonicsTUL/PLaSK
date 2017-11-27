#!/usr/bin/python3
#coding: utf8
from __future__ import print_function, unicode_literals

import sys
import os
from collections import OrderedDict
import textwrap

import yaml

plaskdir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))

fname = sys.argv[1]

cls = sys.argv[2]


def html2rst(text):
    return text\
        .replace(u'<b>', u'\\ **')\
        .replace(u'</b>', u'**\\ ')\
        .replace(u'<i>', u'\\ *')\
        .replace(u'</i>', u'*\\ ')\
        .replace(u'<tt>', u'\\ ``')\
        .replace(u'</tt>', u'``\\ ')\
        .replace(u'<sub>', u'\\ :sub:`')\
        .replace(u'</sub>', u'`\\ ')\
        .replace(u'<sup>', u'\\ :sup:`')\
        .replace(u'</sup>', u'`\\ ')


def browse_attrs(tn, tag, docs, initializers, loaders=None):
    if loaders is None:
        loaders = []
    for attr in tag.get('attrs', []):
        if 'attr' in attr:
            at = attr['attr']
            an = attr.get('var', "{tn}_{an}".format(tn=tn, an=at.replace('-', '_')))
            if an.endswith('#'):
                an = an[:-1]
            if 'default' in attr:
                initializers.append("self.{an} = {ad}".format(an=an, ad=repr(attr['default'])))
                loaders.append("self.{an} = tag.get('{a}', self.{an})".format(an=an, a=at))
            else:
                if attr.get('type', '') == 'geometry path':
                    loaders.append("if '{a}' in tag:".format(a=at))
                    loaders.append("    self.{an} = manager.pth[tag['{a}']]".format(an=an, a=at))
                elif "geometry" in attr.get('type', ''):
                    loaders.append("if '{a}' in tag:".format(a=at))
                    loaders.append("    self.{an} = manager.geo[tag['{a}']]".format(an=an, a=at))
                elif "mesh" in attr.get('type', ''):
                    loaders.append("if '{a}' in tag:".format(a=at))
                    loaders.append("    self.{an} = manager.msh[tag['{a}']]".format(an=an, a=at))
                elif at.endswith('#'):
                    initializers.append("self.{an} = {{}}".format(an=an))
                    loaders.append("for key, val in tag.attrs.items():")
                    loaders.append("    if key.startswith('{}'):".format(at[:-1]))
                    loaders.append("        self.{}[int(key[{}:])] = val".format(an, len(at)-1))
                else:
                    loaders.append("self.{an} = tag['{a}']".format(an=an, a=at))
            if '.' in an:
                docs["{} = object()".format(an.split('.')[0])] = None
            else:
                help = attr.get('help')
                ad = "{} = {}".format(an, repr(attr.get('default', None)))
                if help is not None:
                    docs[ad] = textwrap.wrap(html2rst(help), width=80)
                else:
                    docs[ad] = None
        elif 'group' in attr:
            browse_attrs(tn, attr, docs, initializers, loaders)
    return loaders


def browse_tags(outer, docs, initializers, loaders):
    for tag in outer.get('tags', []):
        if 'tag' in tag:
            tn = tag['tag']
            loaders[tn] = browse_attrs(tn.replace('-', '_'), tag, docs, initializers)
            browse_tags(tag, docs, initializers, loaders)
        if 'bcond' in tag:
            msh = outer.get('mesh')
            if isinstance(msh, list):  msh = msh[0]
            initializers.append("self.{}_boundary = plask.mesh.{}.BoundaryConditions()"
                                .format(tag['bcond'], tag.get('mesh type', msh)))
            loaders[tag['bcond']] = ("self.{}_boundary.read_from_xpl(tag, manager)".format(tag['bcond']),)


try:
	source = yaml.load(open(fname, encoding='utf-8'))
except TypeError:
	source = yaml.load(open(fname))

docs = OrderedDict()
initializers = []
loaders = OrderedDict()

for solver in source:
    if not isinstance(solver, dict): continue
    name = solver.get('solver')
    if name is None or name != cls: continue

    try:
        geom = solver['geometry']
    except KeyError:
        pass
    else:
        loaders['geometry'] = ("self.geometry = tag.getitem(manager.geo, 'ref')", )

    try:
        mesh = solver['mesh']
    except KeyError:
        pass
    else:
        loaders['mesh'] = ("self.mesh = tag.getitem(manager.msh, 'ref')", )

    browse_tags(solver, docs, initializers, loaders)

    print("# coding: utf8")
    print("import plask\n\n")
    print("class {}(plask.Solver):".format(cls))

    for attr, doc in docs.items():
        print("\n    {}".format(attr))
        if doc is not None:
            doc = "\n    ".join(doc)
            print('    """\n    {}\n    """'.format(doc))

    print("\n    def __init__(self, name=''):")
    print("        super({}, self).__init__(name)".format(cls))
    for init in initializers:
        print("        "+init)

    print("\n    def load_xpl(self, xpl, manager):")
    print("        for tag in xpl:")
    el = ''
    for tag, lines in loaders.items():
        print("            {}if tag == '{}':".format(el, tag))
        for line in lines:
            print("                " + line)
        el = 'el'
    print("            else:")
    print('                raise plask.XMLError("{}: Unexpected tag \'{}\'".format(tag, tag.name))')
