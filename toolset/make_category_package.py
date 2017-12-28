#!/usr/bin/python
from __future__ import print_function

import sys
import os

import yaml

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
            source = yaml.load(open_utf8(os.path.join(dirname, 'solvers.yml')))
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
out.write('''"""
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

''' % locals())

for lib, cls in data:
    out.write('   %(lib)s.%(cls)s\n' % locals())

out.write('"""\n')

for lib, cls in data:
    out.write('''\n\ndef %(cls)s(name=''):
    """
    Create %(cls)s solver.
    
    This function automatically loads ``%(lib)s`` submodule and creates
    ``%(lib)s.%(cls)s`` class.
    
    Args:
        name (str): Solver name.
    """
    import %(category)s.%(lib)s
    return %(category)s.%(lib)s.%(cls)s(name)\n''' % locals())
