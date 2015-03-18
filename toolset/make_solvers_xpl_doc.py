#!/usr/bin/python
from __future__ import print_function

XNS = '{http://phys.p.lodz.pl/solvers.xsd}'

import sys
import os

from xml.etree import cElementTree as et

plaskdir = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
docdir = os.path.join(plaskdir, 'doc')

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
    cat = os.path.basename(os.path.dirname(os.path.dirname(dirname)))
    if cat == 'skel': return
    lib = os.path.basename(os.path.dirname(dirname))
    dom = et.parse(os.path.join(dirname, 'solvers.xml'))

    for solver in dom.getroot():
        if solver.tag != XNS+'solver':
            raise ValueError('excpected <solver>, got <{}> instead'.format(solver.tag))

        name = solver.attrib['name']
        cat = solver.attrib.get('category', cat)
        lib = solver.attrib.get('lib', lib)

        try:
            os.makedirs(os.path.join(outdir, cat))
        except OSError:
            pass
        outfile = open(os.path.join(outdir, cat, '{}.{}.rst'.format(lib, name)), 'w')

        def out(*args, **kwargs):
            print(*args, file=outfile, **kwargs)

        def out_text(tag, level):
            text = tag.text
            if text is None: return
            text = html2rst(text.strip())
            if text:
                out()
                for line in text.split('\n'):
                    out('   '*level + line.strip())

        out(name)
        out('-' * (len(name)))
        out('\n.. xml:tag:: <{cat} solver="{name}"> [{name}]\n'.format(**locals()))
        out('   Corresponding Python class: :py:class:`{cat}.{lib}.{name}`.'.format(**locals()))

        out_text(solver, 1)

        out('\n   :attr required name: Solver name.')

        out('\n   .. xml:contents::')

        geom = solver.find(XNS+'geometry').attrib['type']
        out('\n      .. xml:tag:: <geometry> [in {}.{}]'.format(cat, name))
        out('\n         Geometry for use by this solver.')
        out('\n         :attr required ref: Name of a {} geometry defined in the :xml:tag:`<geometry>` section.'
            .format(geom))

        mesh = solver.find(XNS+'mesh')
        if mesh is not None:
            out('\n      .. xml:tag:: <mesh> [in {}.{}]'.format(cat, name))
            out('\n         {} mesh used by this solver.'.format(mesh.attrib['type']))
            out('\n         :attr required ref: Name of a {} mesh defined in the :xml:tag:`<grids>` section.'
                .format(mesh.attrib['type']))

        def write_tags(outer, level=2):
            tags = outer.findall(XNS+'tag') or []
            bconds = outer.findall(XNS+'bcond') or []

            for tag in tags:
                out('\n{}.. xml:tag:: <{}> [in {}.{}]'.format('   '*level, tag.attrib['name'], cat, name))
                out_text(tag, level+1)
                attrs = tag.findall(XNS+'attr')
                if attrs:
                    out()
                    for attr in attrs:
                        doc = ' '.join(line.strip() for line in html2rst(attr.text.strip()).split('\n'))
                        req = 'required' in attr.attrib and attr.attrib['required'].lower() in ('yes', 'true')
                        typ = attr.attrib.get('type', None)
                        unit = attr.attrib.get('unit', None)
                        if typ == 'choice':
                            choices = [ch.text.strip() for ch in attr.findall(XNS+'choice')]
                            if len(choices) == 0:
                                typ = '(choice)'
                            if len(choices) == 1:
                                typ = "('\\ *{}*\\ ')".format(choices[0])
                            elif len(choices) == 2:
                                typ = "('\\ *{}*\\ ' or '\\ *{}*\\ ')".format(choices[0], choices[1])
                            else:
                                typ = "({}, or '\\ *{}*\\ ')".format(
                                    ', '.join("'\\ *{}*\\ '".format(ch) for ch in choices[:-1]), choices[-1])
                        elif typ is not None:
                            if unit is None:
                                typ = '({})'.format(html2rst(typ))
                            else:
                                typ = html2rst('({} [{}])'.format(typ, unit))
                        else:
                            typ = ''
                        out('{}   :attr {}{}: {} {}'.format('   '*level, 'required ' if req else '', attr.attrib['name'], doc, typ))

                write_tags(tag, level+1)

            for bcond in bconds:
                out('\n{}.. xml:tag:: <{}> [in {}.{}]'.format('   '*level, bcond.attrib['name'], cat, name))
                out('\n{}   {} boundary conditions. See subsection :ref:`sec-xpl-Boundary-conditions`.'.format('   '*level, bcond.attrib['label']))
                out_text(bcond, level+1)

        write_tags(solver)

        categories.setdefault(cat, []).append('{}.{}'.format(lib, name))

for dirname, subdirs, files in os.walk(basedir):
    if 'solvers.xml' in files:
        make_rst(dirname)

for cat in categories:
    outfile = open(os.path.join(outdir, '{}.rst'.format(cat)), 'w')

    def out(*args, **kwargs):
        print(*args, file=outfile, **kwargs)

    out('<{}> solvers'.format(cat))
    out('-' * (len(cat)+10))
    out('\n.. toctree::\n')
    for item in categories[cat]:
        out('   {}/{}'.format(cat, item))
