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

from __future__ import print_function

import sys
import os

import yaml

plaskdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(2, os.path.join(plaskdir, 'gui', 'lib'))
import yaml_include  # type: ignore
yaml_include.AddYamlIncludePath(os.path.join(plaskdir, 'plask', 'common'))



source = sys.argv[1]
target = sys.argv[2]

category = os.path.basename(source)
title = category.title()

if category == 'skel': exit()

data = []

cls = None


def open_utf8(*args, **kwargs):
    try:
        return open(*args, encoding='utf-8', **kwargs)
    except TypeError:
        return open(*args, **kwargs)


# Find XML files with solvers configuration
for dirname, _, files in os.walk(source):
    if 'solvers.yml' in files:
        library = os.path.basename(dirname)
        try:
            source = yaml.safe_load(open_utf8(os.path.join(dirname, 'solvers.yml')))
        except:
            continue

        for solver in source:
            if not isinstance(solver, dict): continue
            cls = solver.get('solver')
            if cls is None: continue

            lib = solver.get('lib', library)
            dat = lib, cls
            data.append(dat)

out = open_utf8(os.path.join(target, '__init__.py'), 'w')

if cls is None:
    sys.exit(0)

out.write("# Automatically generated. All your changes will be lost on recompilation!\n\n")
out.write(
    '''"""
%(title)s solvers.

This package combines all %(category)s solvers and helper functions that can
create solver classes without a need to explicitly import the proper binary
module.

Example:

    >>> import %(category)s
    >>> %(category)s.%(cls)s('mysolver')
    <%(category)s.%(lib)s.%(cls)s at 0x42ac2b8>

Solver classes
--------------

.. autosummary::
   :toctree: %(category)s
   :template: solver.rst

''' % locals()
)

for lib, cls in data:
    out.write('   %(lib)s.%(cls)s\n' % locals())

out.write('"""\n')

for lib, cls in data:
    out.write(
        '''\n\ndef %(cls)s(name=''):
    """
    Create %(cls)s solver.

    This function automatically loads ``%(lib)s`` submodule and creates
    :class:`~%(lib)s.%(cls)s` class.

    Args:
        name (str): Solver name.
    """
    import %(category)s.%(lib)s
    return %(category)s.%(lib)s.%(cls)s(name)\n''' % locals()
    )
