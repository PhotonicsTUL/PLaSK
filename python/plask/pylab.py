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
# __doc__ += matplotlib.pylab.__doc__

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

def plot_field(field, levels=16, fill=True, antialiased=False, comp=None, factor=1.0, **kwargs):
    '''Plot scalar real fields as two-dimensional color map'''
    #TODO documentation

    data = field.array

    if type(comp) == str:
        comp = plask.config.axes.index(comp)

    if type(field.mesh) in (plask.mesh.Regular2D, plask.mesh.Rectilinear2D):
        axis0 = field.mesh.axis0
        axis1 = field.mesh.axis1
        if len(data.shape) == 3:
            if comp is None:
                raise TypeError("specify vector component to plot")
            else:
                data = data[:,:,comp]
        data = data.transpose()
    elif type(field.mesh) in (plask.mesh.Regular3D, plask.mesh.Rectilinear3D):
        axes = [ axis for axis in (field.mesh.axis0, field.mesh.axis1, field.mesh.axis2) if len(axis) > 1 ]
        if len(axes) != 2:
            raise TypeError("'plot_field' only accepts 3D mesh with exactly one axis of size 1")
        axis0, axis1 = axes
        if len(data.shape) == 4:
            if comp is None:
                raise TypeError("specify vector component to plot")
            else:
                data = data[:,:,:,comp]
        data = data.reshape((len(axis0), len(axis1))).transpose()
    else:
        raise NotImplementedError("mesh type not supported")

    if 'cmap' in kwargs and type(kwargs['cmap']) == str: # contourf requires that cmap were cmap instance, not a string
        kwargs = kwargs.copy()
        kwargs['cmap'] = get_cmap(kwargs['cmap'])

    if fill:
        result = contourf(axis0, axis1, data*factor, levels, antialiased=antialiased, **kwargs)
    else:
        if 'colors' not in kwargs and 'cmap' not in kwargs:
            result = contour(axis0, axis1, data, levels, colors='k', antialiased=antialiased, **kwargs)
        else:
            result = contour(axis0, axis1, data, levels, antialiased=antialiased, **kwargs)
    return result


def plot_vectors(field, angles='xy', scale_units='xy', **kwargs):
    '''Plot vector field'''
    #TODO documentation

    m = field.mesh

    if type(m) in (plask.mesh.Regular2D, plask.mesh.Rectilinear2D):
        axis0, axis1 = m.axis0, m.axis1
        data = field.array.transpose((1,0,2))
    elif type(m) in (plask.mesh.Regular3D, plask.mesh.Rectilinear3D):
        iaxes = [ iaxis for iaxis in enumerate([m.axis0, m.axis1, m.axis2]) if len(iaxis[1]) > 1 ]
        if len(axes) != 2:
            raise TypeError("'plot_field' only accepts 3D mesh with exactly one axis of size 1")
        (i0, axis0), (i1, axis1) = iaxes
        data = field.array.reshape((len(axis0), len(axis1), field.array.shape[-1]))[:,:,[i0,i1]].transpose((1,0,2,3))
    else:
        raise NotImplementedError("mesh type not supported")

    return quiver(array(axis0), array(axis1), data[:,:,0].real, data[:,:,1].real, angles=angles, scale_units=scale_units, **kwargs)


def plot_stream(field, scale=8.0, color='k', **kwargs):
    '''Plot vector field as a streamlines'''
    #TODO documentation

    m = field.mesh

    if type(m) not in (plask.mesh.Regular2D, plask.mesh.Regular3D):
        raise TypeError("plot_stream can be only used for data obtained for regular mesh")

    if type(m) == plask.mesh.Regular2D:
        axis0, axis1 = m.axis0, m.axis1
        i0, i1 = -2, -1
        data = field.array.transpose((1,0,2))
    elif type(m) == plask.mesh.Regular3D:
        iaxes = [ iaxis for iaxis in enumerate([m.axis0, m.axis1, m.axis2]) if len(iaxis[1]) > 1 ]
        if len(axes) != 2:
            raise TypeError("'plot_stream' only accepts 3D mesh with exactly one axis of size 1")
        (i0, axis0), (i1, axis1) = iaxes
        data = field.array.reshape((len(axis0), len(axis1), field.array.shape[-1]))[:,:,[i0,i1]].transpose((1,0,2))
    else:
        raise NotImplementedError("mesh type not supported")

    if 'linewidth' in kwargs: scale = None

    m0, m1 = meshgrid(array(axis0), array(axis1))
    if scale or color == 'norm':
        norm = sum(data**2, 2)
        norm /= norm.max()
    if color == 'norm':
        color = norm
    if scale:
        return streamplot(m0, m1, data[:,:,0].real, data[:,:,1].real, linewidth=scale*norm, color=color, **kwargs)
    else:
        return streamplot(m0, m1, data[:,:,0].real, data[:,:,1].real, color=color, **kwargs)


def plot_boundary(boundary, mesh, geometry, cmap=None, color='0.75', plane=None, zorder=4, **kwargs):
    '''Plot points of specified boundary'''
    #TODO documentation
    if type(cmap) == str: cmap = get_cmap(cmap)
    if cmap is not None: c = []
    else: c = color
    if isinstance(mesh, plask.mesh.Mesh3D):
        if plane is None:
            raise ValueError("for 3D mesh plane must be specified")
        ax = tuple(i for i,c in enumerate(plask.config.axes) if c in plane)
        if len(ax) != 2:
            raise ValueError("bad plane specified")
    else:
        ax = (0,1)
    x = []
    y = []
    for place, value in boundary:
        points = place(mesh, geometry)
        for i in points:
            x.append(mesh[i][ax[0]])
            y.append(mesh[i][ax[1]])
        if cmap is not None:
            c.extend(len(points) * [value])
    return scatter(x, y, c=c, zorder=zorder, cmap=cmap, **kwargs)


# ### plot_mesh ###

def plot_mesh(mesh, color='0.5', width=1.0, plane=None, set_limits=False, zorder=2):
    '''Plot two-dimensional rectilinear mesh.'''
    #TODO documentation

    axes = matplotlib.pylab.gca()
    lines = []

    if type(mesh) in (plask.mesh.Regular2D, plask.mesh.Rectilinear2D):
        y_min = mesh.axis1[0]; y_max = mesh.axis1[-1]
        for x in mesh.axis0:
            lines.append(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=width, zorder=zorder))
        x_min = mesh.axis0[0]; x_max = mesh.axis0[-1]
        for y in mesh.axis1:
            lines.append(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=width, zorder=zorder))

    elif type(mesh) in (plask.mesh.Regular3D, plask.mesh.Rectilinear3D):
        if plane is None:
            raise ValueError("for 3D mesh plane must be specified")
        axis = tuple((mesh.axis0, mesh.axis1, mesh.axis2)[i] for i,c in enumerate(plask.config.axes) if c in plane)
        if len(axis) != 2:
            raise ValueError("bad plane specified")

        y_min = axis[1][0]; y_max = axis[1][-1]
        for x in axis[0]:
            lines.append(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=width, zorder=zorder))
        x_min = axis[0][0]; x_max = axis[0][-1]
        for y in axis[1]:
            lines.append(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=width, zorder=zorder))


    for line in lines:
        axes.add_line(line)
    if set_limits:
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)

    return lines

# ### plot_geometry ###

_geometry_plotters = {}

def _add_path_Block(patches, trans, box, ax, hmirror, vmirror, color, width, zorder):
    def add_path(bottom):
        lefts = []
        if hmirror:
            if box.lower[ax[0]] == 0.:
                box.lower[ax[0]] = -box.upper[ax[0]]
            elif box.upper[ax[0]] == 0.:
                box.upper[ax[0]] = -box.lower[ax[0]]
            else:
                lefts.append(-box.upper[ax[0]])
        lefts.append(box.lower[ax[0]])
        for left in lefts:
            patches.append(matplotlib.patches.Rectangle((left, bottom),
                                                        box.upper[ax[0]]-box.lower[ax[0]], box.upper[ax[1]]-box.lower[ax[1]],
                                                        ec=color, lw=width, fill=False, zorder=zorder))
    if vmirror:
        if box.lower[ax[1]] == 0.:
            box.lower[ax[1]] = -box.upper[ax[1]]
        elif box.upper[ax[1]] == 0.:
            box.upper[ax[1]] = -box.lower[ax[1]]
        else:
            add_path(-box.upper[ax[1]])

    add_path(box.lower[ax[1]])
_geometry_plotters[plask.geometry.Block2D] = _add_path_Block
_geometry_plotters[plask.geometry.Block3D] = _add_path_Block

def _add_path_Cylinder(patches, trans, box, ax, hmirror, vmirror, color, width, zorder):
    if ax != (0,1):
        _add_path_Block(patches, trans, box, ax, hmirror, vmirror, color, width, zorder)
    else:
        tr = trans.translation
        vecs = [ tr ]
        if hmirror: vecs.append(plask.vec(-tr[0], tr[1]))
        if vmirror: vecs.append(plask.vec(tr[0], -tr[1]))
        if hmirror and vmirror: vecs.append(plask.vec(-tr[0], -tr[1]))
        for vec in vecs:
            patches.append(matplotlib.patches.Circle(vec, trans.item.radius,
                                                    ec=color, lw=width, fill=False, zorder=zorder))
_geometry_plotters[plask.geometry.Cylinder] = _add_path_Cylinder


def plot_geometry(geometry, color='k', width=1.0, plane=None, set_limits=False, zorder=3, mirror=False):
    '''Plot geometry.'''
    #TODO documentation

    axes = matplotlib.pylab.gca()
    patches = []

    if type(geometry) == plask.geometry.Cartesian3D:
        if plane is None:
            raise ValueError("for 3D geometry plane must be specified")
        ax = tuple(i for i,c in enumerate(plask.config.axes) if c in plane)
        if len(ax) != 2:
            raise ValueError("bad plane specified")
        dirs = tuple((("back", "front"), ("left", "right"), ("top", "bottom"))[i] for i in ax)
    else:
        ax = (0,1)
        dirs = (("inner", "outer") if type(geometry) == plask.geometry.Cylindrical2D else ("left", "right"),
                ("top", "bottom"))

    hmirror = mirror and (geometry.borders[dirs[0][0]] == 'mirror' or geometry.borders[dirs[0][1]] == 'mirror' or dirs[0][0] == "inner")
    vmirror = mirror and (geometry.borders[dirs[1][0]] == 'mirror' or geometry.borders[dirs[1][1]] == 'mirror')

    for trans,box in zip(geometry.get_leafs_translations(), geometry.get_leafs_bboxes()):
        _geometry_plotters[type(trans.item)](patches, trans, box, ax, hmirror, vmirror, color, width, zorder)

    for patch in patches:
        axes.add_patch(patch)
    if set_limits:
        box = geometry.bbox
        if hmirror:
            m = max(abs(box.lower[ax[0]]), abs(box.upper[ax[0]]))
            axes.set_xlim(-m, m)
        else:
            axes.set_xlim(box.lower[ax[0]], box.upper[ax[0]])
        if vmirror:
            m = max(abs(box.lower[ax[1]]), abs(box.upper[ax[1]]))
            axes.set_ylim(-m, m)
        else:
            axes.set_ylim(box.lower[ax[1]], box.upper[ax[1]])

    return patches


