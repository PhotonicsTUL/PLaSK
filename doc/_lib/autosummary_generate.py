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
"""
    sphinx.ext.autosummary.generate
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Usable as a library or script to generate automatic RST source files for
    items referred to in autosummary:: directives.

    Each generated RST file contains a single auto*:: directive which
    extracts the docstring of the referred item.

    Example Makefile rule::

       generate:
               sphinx-autogen -o source/generated source/*.rst

    :copyright: Copyright 2007-2011 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
from __future__ import print_function

import os
import re
import sys
import pydoc
import optparse
import codecs

from jinja2 import FileSystemLoader, TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment

from sphinx.ext import autodoc
from sphinx import package_dir
from sphinx.ext.autosummary import import_by_name, get_documenter
from sphinx.jinja2glue import BuiltinTemplateLoader
from sphinx.util.osutil import ensuredir
from sphinx.util.inspect import safe_getattr

# Sphinx bug workaround
try:
    from plask import Solver
except ImportError:
    pass
else:
    boost_class = type(Solver)
    from sphinx.ext.autodoc import AttributeDocumenter
    try:
        AttributeDocumenter.method_types = \
            AttributeDocumenter.method_types + (boost_class,)
    except AttributeError:
        pass


def main(argv=sys.argv):
    usage = """%prog [OPTIONS] SOURCEFILE ..."""
    p = optparse.OptionParser(usage.strip())
    p.add_option("-o", "--output-dir", action="store", type="string",
                 dest="output_dir", default=None,
                 help="Directory to place all output in")
    p.add_option("-s", "--suffix", action="store", type="string",
                 dest="suffix", default="rst",
                 help="Default suffix for files (default: %default)")
    p.add_option("-t", "--templates", action="store", type="string",
                 dest="templates", default=None,
                 help="Custom template directory (default: %default)")
    options, args = p.parse_args(argv[1:])

    if len(args) < 1:
        p.error('no input files given')

    class Sphinx:
        def add_autodocumenter(self, cls):
            autodoc.add_documenter(cls)
        def add_event(self, name):
            pass
        def add_config_value(self, name, default, rebuild):
            pass
    autodoc.setup(Sphinx())

    generate_autosummary_docs(args, options.output_dir,
                              "." + options.suffix,
                              template_dir=options.templates)

def _simple_info(msg):
    print(msg)

def _simple_warn(msg):
    print('WARNING: ' + msg, file=sys.stderr)

# -- Generating output ---------------------------------------------------------

def generate_autosummary_docs(app, sources, output_dir=None, suffix='.rst',
                              warn=_simple_warn, info=_simple_info,
                              base_path=None, builder=None, template_dir=None):

    showed_sources = list(sorted(sources))
    if len(showed_sources) > 20:
        showed_sources = showed_sources[:10] + ['...'] + showed_sources[-10:]
    info('[autosummary] generating autosummary for: %s' %
         ', '.join(showed_sources))

    if output_dir:
        info('[autosummary] writing to %s' % output_dir)

    if base_path is not None:
        sources = [os.path.join(base_path, filename) for filename in sources]

    # create our own templating environment
    template_dirs = [os.path.join(package_dir, 'ext',
                                  'autosummary', 'templates')]
    if builder is not None:
        # allow the user to override the templates
        template_loader = BuiltinTemplateLoader()
        template_loader.init(builder, dirs=template_dirs)
    else:
        if template_dir:
            template_dirs.insert(0, template_dir)
        template_loader = FileSystemLoader(template_dirs)
    template_env = SandboxedEnvironment(loader=template_loader)

    # Define PLaSK-specific tests
    is_provider = lambda item: item.startswith('out') and (item + 'x')[3].isupper()
    is_receiver = lambda item: item.startswith('in') and (item + 'x')[2].isupper()
    template_env.tests['is_provider'] = is_provider
    template_env.tests['is_receiver'] = is_receiver
    template_env.tests['is_other'] = lambda item: not (is_provider(item) or is_receiver(item))

    # read
    items = find_autosummary_in_files(sources)

    # remove possible duplicates
    items = dict([(item, True) for item in items]).keys()

    # keep track of new files
    new_files = []

    # write
    for name, path, template_name in sorted(items, key=lambda it: tuple('' if i is None else i for i in it)):
        if path is None:
            # The corresponding autosummary:: directive did not have
            # a :toctree: option
            continue

        path = output_dir or os.path.abspath(path)
        ensuredir(path)

        try:
            try:
                name, obj, parent, mod_name = import_by_name(name)
            except ValueError:
                name, obj, parent = import_by_name(name)
        except ImportError as e:
            warn('[autosummary] failed to import %r: %s' % (name, e))
            continue

        fn = os.path.join(path, name + suffix)

        # skip it if it exists
        if os.path.isfile(fn):
            continue

        new_files.append(fn)

        f = open(fn, 'w', encoding='utf8')

        try:
            try:
                doc = get_documenter(app, obj, parent)
            except TypeError:
                doc = get_documenter(obj, parent)

            if template_name is not None:
                template = template_env.get_template(template_name)
            else:
                try:
                    template = template_env.get_template('autosummary/%s.rst'
                                                         % doc.objtype)
                except TemplateNotFound:
                    template = template_env.get_template('autosummary/base.rst')

            def get_members(obj, types, include_public=[]):
                if not isinstance(types, (tuple, list)):
                    types = types,
                items = []
                public = []
                for name in dir(obj):
                    try:
                        member = safe_getattr(obj, name)
                        try:
                            documenter = get_documenter(app, member, obj)
                        except TypeError:
                            documenter = get_documenter(member, obj)
                    except AttributeError:
                        continue
                    if documenter.objtype in types:
                        items.append(name)
                        try:
                            docstring = member.__doc__.strip()
                        except:
                            docstring = ''
                        if name in include_public or not (name.startswith('_') or docstring == 'obsolete'):
                            public.append(name)
                return public, items

            ns = {}

            if doc.objtype == 'module':
                ns['members'] = dir(obj)
                ns['functions'], ns['all_functions'] = get_members(obj, 'function')
                ns['classes'], ns['all_classes'] = get_members(obj, 'class')
                ns['exceptions'], ns['all_exceptions'] = get_members(obj, 'exception')
            elif doc.objtype == 'class':
                ns['members'] = dir(obj)
                ns['methods'], ns['all_methods'] = get_members(obj, 'method', ['__call__'])
                ns['attributes'], ns['all_attributes'] = get_members(obj, ('attribute', 'property'))
                ns['classes'], ns['all_classes'] = get_members(obj, 'class')
                ns['static_attributes'] = ns['all_static_attributes'] = []

                ns['docstrings'] = {}

                if 'dtype' in ns['classes']:
                    for l in ns['classes'], ns['all_classes']:
                        del l[l.index('dtype')]
                    ns['static_attributes'].append('dtype')
                    #ns['docstrings']['dtype'] = s[:s.find('.')] + s[s.find('.')]

            parts = name.split('.')
            if doc.objtype in ('method', 'attribute', 'property'):
                mod_name = '.'.join(parts[:-2])
                cls_name = parts[-2]
                obj_name = '.'.join(parts[-2:])
                ns['class'] = cls_name
            elif isinstance(parent, type):
                mod_name = parent.__module__
                obj_name = name[len(mod_name)+1:]
            else:
                mod_name, obj_name = '.'.join(parts[:-1]), parts[-1]

            ns['fullname'] = name
            ns['module'] = mod_name
            ns['objname'] = obj_name
            ns['name'] = parts[-1]

            ns['objtype'] = doc.objtype
            ns['underline'] = len(name) * '='

            rendered = template.render(**ns)
            f.write(rendered)
        finally:
            f.close()

    # descend recursively to new files
    if new_files:
        generate_autosummary_docs(app, new_files, output_dir=output_dir,
                                  suffix=suffix, base_path=base_path,
                                  builder=builder, template_dir=template_dir)


# -- Finding documented entries in files ---------------------------------------

def find_autosummary_in_files(filenames):
    """Find out what items are documented in source/*.rst.

    See `find_autosummary_in_lines`.
    """
    documented = []
    for filename in filenames:
        f = codecs.open(filename, 'r', encoding='utf8', errors='ignore')
        lines = f.read().splitlines()
        documented.extend(find_autosummary_in_lines(lines, filename=filename))
        f.close()
    return documented

def find_autosummary_in_docstring(name, module=None, filename=None):
    """Find out what items are documented in the given object's docstring.

    See `find_autosummary_in_lines`.
    """
    try:
        try:
            real_name, obj, parent, mod_name = import_by_name(name)
        except ValueError:
            real_name, obj, parent = import_by_name(name)
        lines = pydoc.getdoc(obj).splitlines()
        return find_autosummary_in_lines(lines, module=name, filename=filename)
    except AttributeError:
        pass
    except ImportError as e:
        print("Failed to import '%s': %s" % (name, e))
    return []

def find_autosummary_in_lines(lines, module=None, filename=None):
    """Find out what items appear in autosummary:: directives in the
    given lines.

    Returns a list of (name, toctree, template) where *name* is a name
    of an object and *toctree* the :toctree: path of the corresponding
    autosummary directive (relative to the root of the file name), and
    *template* the value of the :template: option. *toctree* and
    *template* ``None`` if the directive does not have the
    corresponding options set.
    """
    autosummary_re = re.compile(r'^(\s*)\.\.\s+autosummary::\s*')
    automodule_re = re.compile(
        r'^\s*\.\.\s+automodule::\s*([A-Za-z0-9_.]+)\s*$')
    module_re = re.compile(
        r'^\s*\.\.\s+(current)?module::\s*([a-zA-Z0-9_.]+)\s*$')
    autosummary_item_re = re.compile(r'^\s+(~?[_a-zA-Z][a-zA-Z0-9_.]*)\s*.*?')
    toctree_arg_re = re.compile(r'^\s+:toctree:\s*(.*?)\s*$')
    template_arg_re = re.compile(r'^\s+:template:\s*(.*?)\s*$')

    documented = []

    toctree = None
    template = None
    current_module = module
    in_autosummary = False
    base_indent = ""

    for line in lines:
        if in_autosummary:
            m = toctree_arg_re.match(line)
            if m:
                toctree = m.group(1)
                if filename:
                    toctree = os.path.join(os.path.dirname(filename),
                                           toctree)
                continue

            m = template_arg_re.match(line)
            if m:
                template = m.group(1).strip()
                continue

            if line.strip().startswith(':'):
                continue # skip options

            m = autosummary_item_re.match(line)
            if m:
                name = m.group(1).strip()
                if name.startswith('~'):
                    name = name[1:]
                if current_module and \
                       not name.startswith(current_module + '.'):
                    name = "%s.%s" % (current_module, name)
                documented.append((name, toctree, template))
                continue

            if not line.strip() or line.startswith(base_indent + " "):
                continue

            in_autosummary = False

        m = autosummary_re.match(line)
        if m:
            in_autosummary = True
            base_indent = m.group(1)
            toctree = None
            template = None
            continue

        m = automodule_re.search(line)
        if m:
            current_module = m.group(1).strip()
            # recurse into the automodule docstring
            documented.extend(find_autosummary_in_docstring(
                current_module, filename=filename))
            continue

        m = module_re.match(line)
        if m:
            current_module = m.group(2)
            continue

    return documented


if __name__ == '__main__':
    import sys
    main(sys.argv)
