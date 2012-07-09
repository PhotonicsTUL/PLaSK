# -*- coding: utf-8 -*-
'''
This is a very simple module for making basic plots in PLaSK using Matplotlib.

In future this module will either be exended or totally removed in favour of GUI.
'''

import pylab
import matplotlib.patches
import matplotlib.lines

import plask

def plotGeometry2D(geometry, axes=None, color='k', width=3.0, set_limits=False, zorder=2):
    if axes is None: axes = pylab.gca()
    for leaf,box in zip(geometry.getLeafsAsTranslations(), geometry.getLeafsBBoxes()):
        if leaf.getMaterial(0.,0.) != plask.material.air:
            axes.add_patch(matplotlib.patches.Rectangle([box.lower[0], box.lower[1]],
                                                        box.upper[0]-box.lower[0], box.upper[1]-box.lower[1],
                                                        ec=color, lw=width, fill=False, zorder=zorder))
    if set_limits:
        box = geometry.bbox
        pad = 0.05 * geometry.bbox_size
        axes.set_xlim(box.lower[0] - pad[0], box.upper[0] + pad[0])
        axes.set_ylim(box.lower[1] - pad[1], box.upper[1] + pad[1])
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
        pad_x = 0.05 * (x_max - x_min)
        pad_y = 0.05 * (y_max - y_min)
        axes.set_xlim(x_min - pad_x, x_max + pad_x)
        axes.set_ylim(y_min - pad_y, y_max + pad_y)
    return axes
