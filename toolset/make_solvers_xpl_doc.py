#!/usr/bin/env python3
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

#coding: utf8
from __future__ import print_function

import sys
import os

from codecs import open

import yaml


plaskdir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
docdir = os.path.join(plaskdir, 'doc')

sys.path.insert(2, os.path.join(plaskdir, 'gui', 'lib'))
import yaml_include
yaml_include.AddYamlIncludePath(os.path.join(plaskdir, 'plask', 'common'))

try:
    basedir = sys.argv[1]
except IndexError:
    basedir = os.path.join(plaskdir, 'solvers')

try:
    outdir = sys.argv[2]
except IndexError:
    outdir = os.path.join(docdir, 'xpl', 'solvers')

categories = {}


def html2rst(text):
    return text\
        .replace('<b>', '\\ **')\
        .replace('</b>', '**\\ ')\
        .replace('<i>', '\\ *')\
        .replace('</i>', '*\\ ')\
        .replace('<tt>', '\\ ``')\
        .replace('</tt>', '``\\ ')\
        .replace('<sub>', '\\ :sub:`')\
        .replace('</sub>', '`\\ ')\
        .replace('<sup>', '\\ :sup:`')\
        .replace('</sup>', '`\\ ')


def make_rst(dirname):
    """Create documentation from solvers.rst"""
    category = os.path.basename(os.path.dirname(dirname))
    library = os.path.basename(dirname)

    source = yaml.safe_load(open(os.path.join(dirname, 'solvers.yml'), encoding='utf8'))

    for solver in source:
        if not isinstance(solver, dict): continue
        name = solver.get('solver')
        if name is None: continue

        cat = solver.get('category', category)
        lib = solver.get('lib', library)

        try:
            os.makedirs(os.path.join(outdir, cat))
        except OSError:
            pass
        outfile = open(os.path.join(outdir, cat, '{}.{}.rst'.format(lib, name)), 'w', encoding='utf8')

        def out(*args, **kwargs):
            print(*args, file=outfile, **kwargs)

        def out_text(text, level, par=False):
            if text is None: return
            indent = len(text)
            text = text.lstrip()
            indent -= len(text)
            text = html2rst(text.rstrip())
            if text:
                if not par: out()
                for line in text.split('\n'):
                    if par: out()
                    if all(c == ' ' for c in line[:indent]):
                        line = line[indent:].rstrip()
                    else:
                        line = line.strip()
                    out('   '*level + line)

        out(name)
        out('-' * (len(name)))
        out('\n.. xml:tag:: <{cat} solver="{name}"> [{name}]\n'.format(**locals()))
        out('   Corresponding Python class: :py:class:`{cat}.{lib}.{name}`.'.format(**locals()))

        out_text(solver.get('help'), 1, True)

        out('\n   :attr required name: Solver name.')

        out('\n   .. xml:contents::')

        try:
            geom = solver['geometry']
        except KeyError:
            pass
        else:
            out('\n      .. xml:tag:: <geometry> [in {}.{}]'.format(cat, name))
            out('\n         Geometry for use by this solver.')
            out('\n         :attr required ref: Name of a {} geometry defined in the :xml:tag:`<geometry>` section.'
                .format(geom))

        try:
            mesh = solver['mesh']
        except KeyError:
            pass
        else:
            if mesh is not None:
                if isinstance(mesh, list): mesh = ', '.join(mesh)
                out('\n      .. xml:tag:: <mesh> [in {}.{}]'.format(cat, name))
                out('\n         {}{} mesh used by this solver.'
                    .format('' if solver.get('need mesh', True) else 'Optional ', mesh))
                out('\n         :attr required ref: Name of a {} mesh defined in the :xml:tag:`<grids>` section.'
                    .format(mesh))

        def write_attrs(attr, level, unit=None):
            if 'attr' in attr:
                doc = html2rst(attr.get('help', '').strip())
                req = attr.get('required')
                typ = attr.get('type')
                unit = attr.get('unit', unit)
                default = attr.get('default')
                if isinstance(default, bool):
                    default = 'yes' if default else 'no'
                elif default is not None:
                    default = str(default)
                if typ == 'choice':
                    choices = [str(ch).strip() for ch in (attr['choices'] + attr.get('other', []))]
                    if len(choices) == 0:
                        typ = 'choice'
                    elif len(choices) == 1:
                        typ = "\\ `{}``\\ ".format(choices[0])
                    elif len(choices) == 2:
                        typ = "\\ ``{}``\\  or \\ ``{}``\\ ".format(choices[0], choices[1])
                    else:
                        typ = "{}, or \\ ``{}``\\ ".format(
                            ', '.join("\\ ``{}``\\ ".format(ch) for ch in choices[:-1]), choices[-1])
                elif typ is not None:
                    if unit is None:
                        typ = '{}'.format(html2rst(typ))
                    else:
                        typ = html2rst('{} [{}]'.format(typ, unit))
                else:
                    typ = ''
                if default is not None:
                    try:
                        float(default)
                    except ValueError:
                        default = "is \\ ``" + default + "``\\ "
                    if typ:
                        typ = typ + ", default {}".format(default)
                    else:
                        typ = "default {}".format(default)
                    if unit is not None:
                        typ += " " + html2rst(unit)
                if typ:
                    typ = '(' + typ + ')'
                doc = ('\n\n' + '   ' * (level+1)).join(doc.split('\n'))
                out('{}   :attr {}{}: {} {}'
                    .format('   ' * level, 'required ' if req else '', attr['attr'], doc, typ))
            else:
                for a in attr['attrs']:
                    write_attrs(a, level, unit)

        def write_tags(outer, level=2):
            for tag in outer:
                if 'tag' in tag:
                    out('\n{}.. xml:tag:: <{}> [in {}.{}]'.format('   '*level, tag['tag'], cat, name))
                    out_text(tag.get('help'), level+1, True)
                    attrs = tag['attrs']
                    if attrs:
                        out()
                        for attr in attrs:
                            write_attrs(attr, level)

                    add_help = tag.get('additional-help')
                    if add_help:
                        out_text('\n' + add_help, level+1, True)

                    write_tags(tag.get('tags', []), level+1)

                elif 'bcond' in tag:
                    out('\n{}.. xml:tag:: <{}> [in {}.{}]'.format('   '*level, tag['bcond'], cat, name))
                    out('\n{}   {} boundary conditions. See subsection :ref:`sec-xpl-Boundary-conditions`.'
                        .format('   '*level, tag['label']))
                    out_text(tag.get('help'), level+1, True)

                    add_help = tag.get('additional-help')
                    if add_help:
                        out_text('\n' + add_help, level+1, True)

        write_tags(solver.get('tags', []))

        categories.setdefault(cat, []).append('{}.{}'.format(lib, name))

        outfile.close()


for dirname, subdirs, files in os.walk(basedir):
    if 'solvers.yml' in files and not os.path.basename(dirname) == 'skel':
        make_rst(dirname)

for cat in categories:
    outfile = open(os.path.join(outdir, '{}.rst'.format(cat)), 'w', encoding='utf8')

    def out(*args, **kwargs):
        print(*args, file=outfile, **kwargs)

    out('<{}> solvers'.format(cat))
    out('-' * (len(cat)+10))
    out('\n.. toctree::\n')
    for item in categories[cat]:
        out('   {}/{}'.format(cat, item))
