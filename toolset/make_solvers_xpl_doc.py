#!/usr/bin/python
#coding: utf8
from __future__ import print_function

import sys
import os

import yaml

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


def make_rst(dirname):
    """Create documentation from solvers.rst"""
    category = os.path.basename(os.path.dirname(dirname))
    library = os.path.basename(dirname)

    source = yaml.load(open(os.path.join(dirname, 'solvers.yml')))

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
        outfile = open(os.path.join(outdir, cat, '{}.{}.rst'.format(lib, name)), 'w')

        def out(*args, **kwargs):
            print(*(a.encode('utf-8') for a in args), file=outfile, **kwargs)

        def out_text(text, level):
            if text is None: return
            text = html2rst(text.strip())
            if text:
                out()
                for line in text.split('\n'):
                    out('   '*level + line.strip())

        out(name)
        out(u'-' * (len(name)))
        out(u'\n.. xml:tag:: <{cat} solver="{name}"> [{name}]\n'.format(**locals()))
        out(u'   Corresponding Python class: :py:class:`{cat}.{lib}.{name}`.'.format(**locals()))

        out_text(solver.get('help'), 1)

        out(u'\n   :attr required name: Solver name.')

        out(u'\n   .. xml:contents::')

        try:
            geom = solver['geometry']
        except KeyError:
            pass
        else:
            out(u'\n      .. xml:tag:: <geometry> [in {}.{}]'.format(cat, name))
            out(u'\n         Geometry for use by this solver.')
            out(u'\n         :attr required ref: Name of a {} geometry defined in the :xml:tag:`<geometry>` section.'
                .format(geom))

        try:
            mesh = solver['mesh']
        except KeyError:
            pass
        else:
            if mesh is not None:
                if isinstance(mesh, list): mesh = ', '.join(mesh)
                out(u'\n      .. xml:tag:: <mesh> [in {}.{}]'.format(cat, name))
                out(u'\n         {} mesh used by this solver.'.format(mesh))
                out(u'\n         :attr required ref: Name of a {} mesh defined in the :xml:tag:`<grids>` section.'
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
                    choices = [str(ch).strip() for ch in attr['choices']]
                    if len(choices) == 0:
                        typ = u'choice'
                    elif len(choices) == 1:
                        typ = u"'\\ *{}*\\ '".format(choices[0])
                    elif len(choices) == 2:
                        typ = u"'\\ *{}*\\ ' or '\\ *{}*\\ '".format(choices[0], choices[1])
                    else:
                        typ = u"{}, or '\\ *{}*\\ '".format(
                            u', '.join(u"'\\ *{}*\\ '".format(ch) for ch in choices[:-1]), choices[-1])
                elif typ is not None:
                    if unit is None:
                        typ = u'{}'.format(html2rst(typ))
                    else:
                        typ = html2rst(u'{} [{}]'.format(typ, unit))
                else:
                    typ = ''
                if default is not None:
                    try:
                        float(default)
                    except ValueError:
                        default = u"is '\\ *" + default + u"*\\ '"
                    if typ:
                        typ = typ + u", default {}".format(default)
                    else:
                        typ = u"default {}".format(default)
                    if unit is not None:
                        typ += u" " + html2rst(unit)
                if typ:
                    typ = u'(' + typ + u')'
                doc = (u'\n\n' + u'   ' * level).join(doc.split(u'\n'))
                out(u'{}   :attr {}{}: {} {}'
                    .format(u'   ' * (level+1), u'required ' if req else u'', attr['attr'], doc, typ))
            else:
                for a in attr['attrs']:
                    write_attrs(a, level, unit)

        def write_tags(outer, level=2):
            for tag in outer:
                if 'tag' in tag:
                    out(u'\n{}.. xml:tag:: <{}> [in {}.{}]'.format('   '*level, tag['tag'], cat, name))
                    out_text(tag.get('help'), level+1)
                    attrs = tag['attrs']
                    if attrs:
                        out()
                        for attr in attrs:
                            write_attrs(attr, level)

                    write_tags(tag.get('tags', []), level+1)

                elif 'bcond' in tag:
                    out(u'\n{}.. xml:tag:: <{}> [in {}.{}]'.format('   '*level, tag['bcond'], cat, name))
                    out(u'\n{}   {} boundary conditions. See subsection :ref:`sec-xpl-Boundary-conditions`.'
                        .format('   '*level, tag['label']))
                    out_text(tag.get('help'), level+1)

        write_tags(solver.get('tags', []))

        categories.setdefault(cat, []).append('{}.{}'.format(lib, name))

for dirname, subdirs, files in os.walk(basedir):
    if 'solvers.yml' in files and not os.path.basename(dirname) == 'skel':
        make_rst(dirname)

for cat in categories:
    outfile = open(os.path.join(outdir, '{}.rst'.format(cat)), 'w')

    def out(*args, **kwargs):
        print(*args, file=outfile, **kwargs)

    out(u'<{}> solvers'.format(cat))
    out(u'-' * (len(cat)+10))
    out(u'\n.. toctree::\n')
    for item in categories[cat]:
        out('   {}/{}'.format(cat, item))
