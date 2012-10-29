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

import matplotlib.colors
import matplotlib.lines
import matplotlib.patches

import matplotlib.pylab
from matplotlib.pylab import *
__doc__ += matplotlib.pylab.__doc__

import plask

def plot_field(field, levels=None, **kwargs):
    '''Plot scalar real fields as two-dimensional color map'''
    #TODO documentation
    if levels is None:
        if type(field.mesh == plask.mesh.Regular2D):
            result = imshow(field.array, origin='lower', extent=[field.mesh.axis0[0], field.mesh.axis0[-1], field.mesh.axis1[0], field.mesh.axis1[-1]], **kwargs)
        else:
            if 'aspect' in kwargs:
                kwargs = kwargs.copy()
                set_aspect(kwargs.pop('aspect'))
            adata = field.array
            data = adata[:-1,:-1]; data += adata[1:,:-1]; data += adata[:-1,1:]; data += adata[1:,1:]; data *= 0.25
            del adata
            result = pcolor(array(field.mesh.axis0), array(field.mesh.axis1), data, **kwargs)
    else:
        if 'cmap' in kwargs and type(kwargs['cmap']) == str: # contourf requires that cmap were cmap instance, not a string
            kwargs = kwargs.copy()
            kwargs['cmap'] = get_cmap(kwargs['cmap'])
        result = contourf(field.mesh.axis0, field.mesh.axis1, field.array, levels, antialiased=True, **kwargs)
    return result


def plot_geometry(geometry, color='k', width=1.0, set_limits=False, zorder=3, mirror=False):
    '''Plot geometry.'''
    #TODO documentation
    axes = matplotlib.pylab.gca()
    patches = []
    for leaf,box in zip(geometry.get_leafs_translations(), geometry.get_leafs_bboxes()):
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


def plot_mesh(mesh, color='0.5', width=1.0, set_limits=False, zorder=2):
    '''Plot two-dimensional rectilinear mesh.'''
    #TODO documentation
    axes = matplotlib.pylab.gca()
    lines = []
    if type(mesh) in [plask.mesh.Regular2D, plask.mesh.Rectilinear2D]:
        y_min = mesh.axis1[0]; y_max = mesh.axis1[-1]
        for x in mesh.axis0:
            lines.append(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=width, zorder=zorder))
        x_min = mesh.axis0[0]; x_max = mesh.axis0[-1]
        for y in mesh.axis1:
            lines.append(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=width, zorder=zorder))
    for line in lines:
        axes.add_line(line)
    if set_limits:
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)

    # return lines


def plot_boundary(boundary, mesh, cmap=None, color='0.75', zorder=4, **kwargs):
    '''Plot points of specified boundary'''
    #TODO documentation
    if type(cmap) == str: cmap = get_cmap(cmap)
    if cmap is not None: c = []
    else: c = color
    x = []
    y = []
    for place, value in boundary:
        points = place(mesh)
        for i in points:
            x.append(mesh[i][0])
            y.append(mesh[i][1])
        if cmap is not None:
            c.extend(len(points) * [value])
    return scatter(x, y, c=c, zorder=zorder, cmap=cmap, **kwargs)


def plot_material_param(geometry, param, axes=None, mirror=False, **kwargs):
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
    points = grid.get_midpoints_mesh()
    data = array([ param(geometry.get_material(p)) for p in points ]).reshape((len(points.axis1), len(points.axis0)))
    return pcolor(array(grid.axis0), array(grid.axis1), data, **kwargs)

