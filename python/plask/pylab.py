# -*- coding: utf-8 -*-
'''
This is an interface to Matplotlib and pylab.

It contains some helper functions helping with plotting PLaSK
input/output.

Additional functions defined here are:

TODO


Below there follows the documentation of Matplotlib pylab:

----------------------------------------------------------------------
'''

import matplotlib.pylab
from matplotlib.pylab import *
__doc__ += matplotlib.pylab.__doc__

import plask

def plotField2D(field, **kwargs):
    '''Plot scalar real fields using pylab.imshow.'''
    #TODO documentation
    return imshow(field.array, origin='lower', extent=[field.mesh.axis0[0], field.mesh.axis0[-1], field.mesh.axis1[0], field.mesh.axis1[-1]], **kwargs)


def plotGeometry2D(geometry, color='k', width=1.0, set_limits=False, zorder=3, mirror=False):
    '''Plot two-dimensional geometry.'''
    #TODO documentation

    import matplotlib.patches
    axes = matplotlib.pylab.gca()

    patches = []
    for leaf,box in zip(geometry.getLeafsAsTranslations(), geometry.getLeafsBBoxes()):
        #TODO other shapes than rectangles
        def add_path(bottom):
            lefts = [box.lower[0]]
            if mirror and (type(geometry) == plask.geometry.Cylindrical2D or \
                           geometry.borders['left'] == 'mirror' or geometry.borders['right'] == 'mirror'):
                lefts.append(-box.upper[0])
            for left in lefts:
                patches.append(matplotlib.patches.Rectangle([left, bottom],
                                                            box.upper[0]-box.lower[0], box.upper[1]-box.lower[1],
                                                            ec=color, lw=width, fill=False, zorder=zorder))
        add_path(box.lower[1])
        if mirror and (geometry.borders['top'] == 'mirror' or geometry.borders['bottom'] == 'mirror'):
            add_path(-box.upper[1])

    for patch in patches:
        axes.add_patch(patch)

    if set_limits:
        box = geometry.bbox
        axes.set_xlim(box.lower[0], box.upper[0])
        axes.set_ylim(box.lower[1], box.upper[1])

    # return patches


def plotMesh2D(mesh, color='0.5', width=1.0, set_limits=False, mirror=(False, False), zorder=2):
    '''Plot two-dimensional rectilinear mesh.'''
    #TODO documentation

    mirx = mirror[0]
    miry = mirror[1]

    import matplotlib.lines
    axes = matplotlib.pylab.gca()

    lines = []
    if type(mesh) in [plask.mesh.Regular2D, plask.mesh.Rectilinear2D]:
        y_min = mesh.axis1[0]; y_max = mesh.axis1[-1]
        if miry:
            y_min = -y_max
        for x in mesh.axis0:
            lines.append(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=width, zorder=zorder))
            if mirx:
                lines.append(matplotlib.lines.Line2D([-x,-x], [y_min,y_max], color=color, lw=width, zorder=zorder))
        x_min = mesh.axis0[0]; x_max = mesh.axis0[-1]
        if mirx: x_min = -x_max
        for y in mesh.axis1:
            lines.append(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=width, zorder=zorder))
            if miry:
                lines.append(matplotlib.lines.Line2D([x_min,x_max], [-y,-y], color=color, lw=width, zorder=zorder))

    for line in lines:
        axes.add_line(line)

    if set_limits:
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)

    # return lines


def plotMaterialParam2D(geometry, param, axes=None, mirror=False, **kwargs):
    '''Plot selected material parameter as color map'''
    #TODO documentation

    if axes is None: axes = matplotlib.pylab.gca()

    if type(param) == str:
        param = eval('lambda m: m.' + param)

    #TODO for different shapes of leafs, plot it somehow better (how? make patches instead of pcolor?)
    grid = plask.mesh.Rectilinear2D.SimpleGenerator()(geometry.child)
    grid.ordering = '10'
    if mirror:
        if type(geometry) == plask.geometry.Cylindrical2D or \
           geometry.borders['left'] == 'mirror' or geometry.borders['right'] == 'mirror':
            ax = array(grid.axis0)
            grid.axis0 = concatenate((-ax, ax))
        if geometry.borders['top'] == 'mirror' or geometry.borders['bottom'] == 'mirror':
            ax = array(grid.axis1)
            grid.axis1 = concatenate((-ax, ax))
    points = grid.getMidpointsMesh()
    data = array([ param(geometry.getMaterial(p)) for p in points ]).reshape((len(points.axis1), len(points.axis0)))

    return pcolor(array(grid.axis0), array(grid.axis1), data, **kwargs)
