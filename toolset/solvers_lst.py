import sys
import os.path as pth

out = open(pth.join(sys.argv[1], sys.argv[2], '__init__.py'), 'w')

category = sys.argv[2].split('/')[1]
title = category.title()

data = []
for l in open(pth.join(sys.argv[1], sys.argv[2], 'solvers.lst'), 'r'):
    line = l.strip()
    if not line: continue
    data.append(line.split('.'))

lib0, cls0 = data[0]

out.write("# Automatically generated. All your changes will be lost on recompilation!\n\n")
out.write("""'''
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

""" % locals())

for lib,cls in data:
    out.write("   %(lib)s.%(cls)s\n" % locals())

out.write("'''\n")

for lib,cls in data:
    out.write("""\ndef %(cls)s(name=''):
    '''Create %(cls)s solver.

    This function automatically loads ``%(lib)s`` submodule and creates
    ``%(lib)s.%(cls)s`` class.

    Args:
        name (str): Solver name.
    '''
    import %(category)s.%(lib)s
    return %(category)s.%(lib)s.%(cls)s(name)\n""" % locals())
