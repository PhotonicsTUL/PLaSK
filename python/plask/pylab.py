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

# Easier rc handling. Make conditional on matplotlib version if I manage to introduce it there
if True:
    class _subRC(object):
        def __init__(self, group):
            self.__dict__['_group'] = group

        def __setattr__(self, attr, value):
            name = _Rc.aliases.get(attr) or attr
            key = self._group + '.' + name
            if key not in rcParams:
                raise KeyError('Unrecognized key "%s"' % key)
            rcParams[key] = value

        def __getattr__(self, attr):
            name = _Rc.aliases.get(attr) or attr
            newgroup = self._group + '.' + name
            return _subRC(newgroup)

    class _Rc(object):
        """
        Set the current rc params.  There are two alternative ways of using
        this object.  One is to call it like a function::

        rc(group, **kwargs)

        Group is the grouping for the rc, eg. for ``lines.linewidth``
        the group is ``lines``, for ``axes.facecolor``, the group is ``axes``,
        and so on.  Group may also be a list or tuple of group names,
        eg. (*xtick*, *ytick*).  *kwargs* is a dictionary attribute name/value
        pairs, eg::

        rc('lines', linewidth=2, color='r')

        sets the current rc params and is equivalent to::

        rcParams['lines.linewidth'] = 2
        rcParams['lines.color'] = 'r'

        The following aliases are available to save typing for interactive
        users:

        =====   =================
        Alias   Property
        =====   =================
        'lw'    'linewidth'
        'ls'    'linestyle'
        'c'     'color'
        'fc'    'facecolor'
        'ec'    'edgecolor'
        'mew'   'markeredgewidth'
        'aa'    'antialiased'
        'sans'  'sans-serif'
        =====   =================

        Thus you could abbreviate the above rc command as::

            rc('lines', lw=2, c='r')


        Note you can use python's kwargs dictionary facility to store
        dictionaries of default parameters.  Eg, you can customize the
        font rc as follows::

        font = {'family' : 'monospace',
                'weight' : 'bold',
                'size'   : 'larger'}

        rc('font', **font)  # pass in the font dict as kwargs

        This enables you to easily switch between several configurations.
        Use :func:`~matplotlib.pyplot.rcdefaults` to restore the default
        rc params after changes.

        Another way of using this object is to use the python syntax like::

        rc.figure.subplot.top = 0.9

        which is equivalent to::

        rc('figure.subplot', top=0.9)
        """

        aliases = {
            'lw'  : 'linewidth',
            'ls'  : 'linestyle',
            'c'   : 'color',
            'fc'  : 'facecolor',
            'ec'  : 'edgecolor',
            'mew' : 'markeredgewidth',
            'aa'  : 'antialiased',
            'sans': 'sans-serif'
        }

        def __call__(self, group, **kwargs):
            if matplotlib.is_string_like(group):
                group = (group,)
            for g in group:
                for k,v in kwargs.items():
                    name = _Rc.aliases.get(k) or k
                    key = '%s.%s' % (g, name)
                    if key not in rcParams:
                        raise KeyError('Unrecognized key "%s" for group "%s" and name "%s"' %
                                    (key, g, name))
                    rcParams[key] = v

        def __setattr__(self, attr, value):
            key = _Rc.aliases.get(attr) or attr
            if key not in rcParams:
                raise KeyError('Unrecognized key "%s"' % key)
            rcParams[key] = value

        def __getattribute__(self, attr):
            if attr[:2] != '__':
                return _subRC(attr)
            else:
                raise AttributeError

    rc = _Rc()
    _Rc.__name__ = 'rc'


import plask

def plot_field(field, levels=16, antialiased=True, **kwargs):
    '''Plot scalar real fields as two-dimensional color map'''
    #TODO documentation

    if type(field.mesh) in (plask.mesh.Regular2D, plask.mesh.Rectilinear2D):
        axis0 = field.mesh.axis0
        axis1 = field.mesh.axis1
        data = field.array
    elif type(field.mesh) in (plask.mesh.Regular3D, plask.mesh.Rectilinear3D):
        axes = [ axis for i,axis in (field.mesh.axis0, field.mesh.axis1, field.mesh.axis2) if len(axis) > 1 ]
        if len(axes) != 2:
            raise TypeError("'plot_field' only accepts 3D mesh with exactly one axis of size 1")
        axis0, axis1 = axes
        data = field.array.reshape((len(axis1), len(axis0)))
    else:
        raise NotImplementedError("mesh type not supported")

    if 'cmap' in kwargs and type(kwargs['cmap']) == str: # contourf requires that cmap were cmap instance, not a string
        kwargs = kwargs.copy()
        kwargs['cmap'] = get_cmap(kwargs['cmap'])

    result = contourf(axis0, axis1, data, levels, antialiased=antialiased, **kwargs)
    return result


def plot_vectors(field, color='w', angles='xy', scale_units='xy', **kwargs):
    '''Plot vector field'''
    #TODO documentation
    m = field.mesh
    quiver(m.axis0, m.axis1, field.array[:,:,0], field.array[:,:,1],
           color=color, angles=angles, scale_units=scale_units, **kwargs)


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
    points = grid.get_midpoints()
    data = array([ param(geometry.get_material(p)) for p in points ]).reshape((len(points.axis1), len(points.axis0)))
    return pcolor(array(grid.axis0), array(grid.axis1), data, antialiased=True, edgecolors='none', **kwargs)

