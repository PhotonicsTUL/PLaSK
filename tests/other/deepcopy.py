# coding: utf8
from __future__ import print_function

from copy import copy, deepcopy

def show_tree(obj, level='', mark=''):
    print(u"{}{}{} at (0x{:x})\t0x{:x}".format(level, mark, type(obj).__name__, hash(obj), id(obj)).expandtabs(40))
    level = level + (u'  ' if mark == u' └' else u' │' if mark == u' ├' else u'' )
    try:
        n = len(obj) - 1
    except TypeError:
        n = 0
    for i, o in enumerate(obj):
        show_tree(o, level, (u' └' if i == n else u' ├'))

rect = geometry.Rectangle(2, 1, 'GaAs')
triangle = geometry.Triangle((2,0), (0, 1), 'AlAs')
circle = geometry.Circle(1, 'AlN')
clip = geometry.Clip2D(circle, bottom=0)

overlap = geometry.Align2D()
overlap.append(geometry.Rectangle(2, 1, 'AlOx'), 0, 0)
overlap.append(triangle, 0, 0)

stack = geometry.Stack2D()
stack.prepend(circle)
stack.prepend(clip)
stack.prepend(overlap)
stack.prepend(rect)

memo = {}

geo = geometry.Cartesian2D(stack)
geo1 = copy(geo)
geo2 = deepcopy(geo, memo)

show_tree(geo)
print()
show_tree(geo1)
print()
show_tree(geo2)
print()
for k,v in memo.items():
    print('0x{:x}: {}'.format(k,v))

if __name__ == '__main__':
    plot_geometry(geo2, margin=0.02, fill=True)
    show()
