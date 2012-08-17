# -*- coding: utf-8 -*-
'''
This is a very simple solver for making basic plots in PLaSK using Matplotlib.

In future this solver will either be exended or totally removed in favour of GUI.
'''

import pylab
import matplotlib.patches
import matplotlib.lines

import plask

def plotGeometry2D(geometry, axes=None, color='k', width=1.0, set_limits=False, zorder=3, mirror=False):
    if axes is None: axes = pylab.gca()
    for leaf,box in zip(geometry.getLeafsAsTranslations(), geometry.getLeafsBBoxes()):
        def add_path(bottom):
            lefts = [box.lower[0]]
            if mirror and (geometry.borders['left'] == 'mirror' or geometry.borders['right']): lefts.append(-box.upper[0])
            for left in lefts:
                axes.add_patch(matplotlib.patches.Rectangle([left, bottom],
                                                            box.upper[0]-box.lower[0], box.upper[1]-box.lower[1],
                                                            ec=color, lw=width, fill=False, zorder=zorder))
        add_path(box.lower[1])
        if mirror and (geometry.borders['top'] == 'bottom' or geometry.borders['right']):
            add_path(-box.upper[1])

    if set_limits:
        box = geometry.bbox
        axes.set_xlim(box.lower[0], box.upper[0])
        axes.set_ylim(box.lower[1], box.upper[1])
    return axes

def plotRectilinearMesh2D(mesh, axes=None, color='0.5', width=1.0, set_limits=False, zorder=2):
    if axes is None: axes = pylab.gca()
    y_min = mesh.axis1[0]; y_max = mesh.axis1[-1]
    for x in mesh.axis0:
        axes.add_line(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=width, zorder=zorder))
    x_min = mesh.axis0[0]; x_max = mesh.axis0[-1]
    for y in mesh.axis1:
        axes.add_line(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=width, zorder=zorder))
    if set_limits:
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
    return axes
