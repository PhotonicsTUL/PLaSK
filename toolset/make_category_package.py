#!/usr/bin/python
from __future__ import print_function

import sys
import os

from lxml import etree as et

source = sys.argv[1]
target = sys.argv[2]

category = os.path.basename(source)
title = category.title()

if category == 'skel': exit()

data = []

# Find XML files with solvers configuration
for dirname, _, files in os.walk(source):
    if 'solvers.xml' in files:
        library = os.path.basename(dirname)
        try:
            dom = et.parse(os.path.join(dirname, 'solvers.xml'))
        except et.XMLSyntaxError:
            continue

        root = dom.getroot()
        xns = root.nsmap.get(None, '')
        if xns: xns = '{'+xns+'}'

        for solver in root.findall(xns+'solver'):
            cat = solver.attrib.get('category', category)
            dat = solver.attrib.get('lib', library), solver.attrib['name'], \
                  solver.attrib.get('obsolete', '').lower() not in ('yes', 'true', '1')
            data.append(dat)

out = open(os.path.join(target, '__init__.py'), 'w')

for lib0,cls0,ok0 in data:
    if not ok0: break

out.write("# Automatically generated. All your changes will be lost on recompilation!\n\n")
out.write('''"""
%(title)s solvers.

This package combines all %(category)s solvers and helper functions that can
create solver classes without a need to explicitly import the proper binary
module.

Example:

    >>> import %(category)s
    >>> %(category)s.%(cls0)s('mysolver')
    <%(category)s.%(lib0)s.%(cls0)s at 0x42ac2b8>

Solver classes
--------------

.. autosummary::
   :toctree: %(category)s
   :template: solver.rst

''' % locals())

for lib,cls,ok in data:
    if ok:
        out.write('   %(lib)s.%(cls)s\n' % locals())

out.write('"""\n')

for lib,cls,ok in data:
  if ok:
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
  else:
    out.write('''\n\ndef %(cls)s(name=''):
    "obsolete"
    import %(category)s.%(lib)s
    return %(category)s.%(lib)s.%(cls)s(name)\n''' % locals())
