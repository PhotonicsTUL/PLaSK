# -*- coding: utf-8 -*-
"""
This is an interface to Matplotlib and pylab.

It contains some helper functions helping with plotting PLaSK
input/output.

Additional functions defined here are:

TODO

"""

import os as _os

import matplotlib
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.artist

try:
    backend = _os.environ['MATPLOTLIB_BACKEND']
except KeyError:
    backend = matplotlib.rcParams['backend']
else:
    matplotlib.use(backend)

# Specify Qt4 API v2 while it is not too late
if backend == 'Qt4Agg' and matplotlib.rcParams['backend.qt4'] == 'PyQt4':
    try:
        import sip
        for n in ("QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"): sip.setapi(n, 2)
    except:
        pass

import matplotlib.pylab
from matplotlib.pylab import *
# __doc__ += matplotlib.pylab.__doc__

# Easier rc handling. Make conditional on matplotlib version if I manage to introduce it there
if True:
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

        Another way of using this object is to use the Python syntax like::

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

        class Group(object):
            def __init__(self, group):
                self.__dict__['_group'] = group

            def __setattr__(self, attr, value):
                name = _Rc.aliases.get(attr, attr)
                key = self._group + '.' + name
                if key not in rcParams:
                    raise KeyError(u'Unrecognized key "{}"'.format(key))
                rcParams[key] = value

            def __getattr__(self, attr):
                name = _Rc.aliases.get(attr) or attr
                newgroup = self._group + '.' + name
                if newgroup in rcParams:
                    return rcParams[newgroup]
                else:
                    return _Rc.Group(newgroup)

        def __call__(self, group, **kwargs):
            if matplotlib.is_string_like(group):
                group = (group,)
            for g in group:
                for k,v in kwargs.items():
                    name = _Rc.aliases.get(k) or k
                    key = '{}.{}'.format(g, name)
                    if key not in rcParams:
                        raise KeyError(
                            'Unrecognized key "{0}" for group "{1}" and name "{2}"'.format(key, g, name))
                    rcParams[key] = v

        def __setattr__(self, attr, value):
            key = _Rc.aliases.get(attr) or attr
            if key not in rcParams:
                raise KeyError(u'Unrecognized key "{}"'.format(key))
            rcParams[key] = value

        def __getattribute__(self, attr):
            if attr[:2] != '__':
                return _Rc.Group(attr)
            else:
                raise AttributeError

    rc = _Rc()
    _Rc.__name__ = 'rc'


import plask


def _get_2d_axes(plane):
    if plane is None:
        raise ValueError("Must specify plane for 3D projection")
    axes = tuple(int(a) if a in ('0', '1', '2') else 'ltv'.find(a) if a in 'ltv' else plask.config.axes.find(a) for a in plane)
    if -1 in axes:
        raise ValueError("Wrong plane '{}'".format(plane))
    if axes[0] == axes[1]:
        raise ValueError("Repeated axis in plane '{}'".format(plane))
    return axes


def _get_component(comp, total):
    if type(comp) == str:
        if total == 4: # tensor
            try:
                a = plask.config.axes
                values = (a[0]+a[0], a[1]+a[1], a[2]+a[2], a[0]+a[1], a[1]+a[0])
                comp = min(values.index(comp), 3)
            except ValueError:
                comp = min(('ll', 'tt', 'vv', 'lt', 'tl').index(comp), 3)
        else:
            if comp in ('long', 'tran', 'vert'):
                comp = comp[0]
            try:
                if plask.config.axes == 'long,tran,vert':
                    raise ValueError
                comp = plask.config.axes.index(comp)
            except ValueError:
                comp = 'ltv'.index(comp)
    if total == 2:
        comp = max(comp-1, 0)
    return comp


def plot_field(field, levels=16, plane=None, fill=True, antialiased=False, comp=None, **kwargs):
    """Plot scalar real fields as two-dimensional color map"""
    #TODO documentation

    data = field.array

    if isinstance(field.mesh, plask.mesh.Rectangular2D):
        ax = 0, 1
        xaxis = field.mesh.axis0
        yaxis = field.mesh.axis1
        if len(data.shape) == 3:
            if comp is None:
                raise TypeError("Specify vector component to plot")
            else:
                comp = _get_component(comp, data.shape[2])
                data = data[:,:,comp]
        data = data.transpose()
    elif isinstance(field.mesh, plask.mesh.Rectangular3D):
        ax = _get_2d_axes(plane)
        if data.shape[3-sum(ax)] != 1:
            raise ValueError("Field mesh must have dimension {} equal to 1".format(3-sum(ax)))
        xaxis, yaxis = ((field.mesh.axis0, field.mesh.axis1, field.mesh.axis2)[i] for i in ax)
        if len(data.shape) == 4:
            if comp is None:
                raise TypeError("Specify vector component to plot")
            else:
                comp = _get_component(comp, data.shape[3])
                data = data[:,:,:,comp]
        if ax[0] < ax[1]:
            data = data.reshape((len(xaxis), len(yaxis))).transpose()
        else:
            data = data.reshape((len(yaxis), len(xaxis)))
    else:
        raise NotImplementedError("Mesh type not supported")

    if 'cmap' in kwargs and type(kwargs['cmap']) == str: # contourf requires that cmap is a cmap instance, not a string
        kwargs = kwargs.copy()
        kwargs['cmap'] = get_cmap(kwargs['cmap'])

    if fill:
        result = contourf(xaxis, yaxis, data, levels, antialiased=antialiased, **kwargs)
    else:
        if 'colors' not in kwargs and 'cmap' not in kwargs:
            result = contour(xaxis, yaxis, data, levels, colors='k', antialiased=antialiased, **kwargs)
        else:
            result = contour(xaxis, yaxis, data, levels, antialiased=antialiased, **kwargs)

    axes = matplotlib.pylab.gca()
    if ax[0] > ax[1] and not axes.yaxis_inverted():
        axes.invert_yaxis()
    xlabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + ax[0]]))
    ylabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + ax[1]]))

    return result


def plot_profile(field, comp=None, swap_axes=False, **kwargs):
    """Plot scalar real fields as two-dimensional color map"""
    #TODO documentation

    data = field.array

    if isinstance(field.mesh, plask.mesh.Rectangular2D):
        if len(field.mesh.axis0) != 1 and len(field.mesh.axis1) == 1:
            ax = 1
            axis = field.mesh.axis0
        elif len(field.mesh.axis0) == 1 and len(field.mesh.axis1) != 1:
            ax = 2
            axis = field.mesh.axis1
        else:
            raise ValueError("Exactly one mesh dimension must be different than 1")
        if len(data.shape) == 3:
            if comp is None:
                raise TypeError("Specify vector component to plot")
            else:
                comp = _get_component(comp, data.shape[2])
                data = data[:,:,comp]
    elif isinstance(field.mesh, plask.mesh.Rectangular3D):
        if len(field.mesh.axis0) != 1 and len(field.mesh.axis1) == 1 and len(field.mesh.axis2) == 1:
            ax = 0
            axis = field.mesh.axis0
        elif len(field.mesh.axis0) == 1 and len(field.mesh.axis1) != 1 and len(field.mesh.axis2) == 1:
            ax = 1
            axis = field.mesh.axis1
        elif len(field.mesh.axis0) == 1 and len(field.mesh.axis1) == 1 and len(field.mesh.axis2) != 1:
            ax = 2
            axis = field.mesh.axis2
        else:
            raise ValueError("Exactly one mesh dimension must be different than 1")
        if len(data.shape) == 4:
            if comp is None:
                raise TypeError("Specify vector component to plot")
            else:
                comp = _get_component(comp, data.shape[3])
                data = data[:,:,:,comp]
    else:
        raise NotImplementedError("Mesh type not supported")

    if swap_axes:
        ylabel(u"${}$ [µm]".format(plask.config.axes[ax]))
        return plot(data.ravel(), axis, **kwargs)
    else:
        xlabel(u"${}$ [µm]".format(plask.config.axes[ax]))
        return plot(axis, data.ravel(), **kwargs)


def plot_vectors(field, plane=None, angles='xy', scale_units='xy', **kwargs):
    """Plot vector field"""
    #TODO documentation

    m = field.mesh

    if isinstance(m, plask.mesh.Rectangular2D):
        ix, iy = 0, 1
        xaxis, yaxis = m.axis0, m.axis1
        data = field.array.transpose((1,0,2))
    elif isinstance(m, plask.mesh.Rectangular3D):
        ix, iy = _get_2d_axes(plane)
        if data.shape[3-ix-iy] != 1:
            raise ValueError("Field mesh must have dimension {} equal to 1".format(3-ix-iy))
        xaxis, yaxis = ((field.mesh.axis0, field.mesh.axis1, field.mesh.axis2)[i] for i in (ix,iy))
        if ix < iy:
            data = field.array.reshape((len(xaxis), len(yaxis), field.array.shape[-1]))[:,:,[ix,iy]].transpose((1,0,2))
        else:
            data = field.array.reshape((len(yaxis), len(xaxis), field.array.shape[-1]))[:,:,[ix,iy]]
    else:
        raise NotImplementedError("mesh type not supported")

    axes = matplotlib.pylab.gca()
    if ix > iy and not axes.yaxis_inverted():
        axes.invert_yaxis()
    xlabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + ix]))
    ylabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + iy]))

    return quiver(array(xaxis), array(yaxis), data[:,:,0].real, data[:,:,1].real, angles=angles, scale_units=scale_units, **kwargs)


def plot_stream(field, plane=None, scale=8.0, color='k', **kwargs):
    """Plot vector field as a streamlines"""
    #TODO documentation

    m = field.mesh

    if isinstance(m, plask.mesh.Rectangular2D):
        if type(m.axis0) != plask.mesh.Regular or type(m.axis1) != plask.mesh.Regular:
            raise TypeError("plot_stream can be only used for data obtained for rectangular mesh with regular axes")
        xaxis, yaxis = m.axis0, m.axis1
        ix, iy = -2, -1
        data = field.array.transpose((1,0,2))
    elif isinstance(m, plask.mesh.Rectangular3D):
        if type(m.axis0) != plask.mesh.Regular or type(m.axis1) != plask.mesh.Regular or type(m.axis2) != plask.mesh.Regular:
            raise TypeError("plot_stream can be only used for data obtained for rectangular mesh with regular axes")
        ix, iy = _get_2d_axes(plane)
        xaxis, yaxis = ((field.mesh.axis0, field.mesh.axis1, field.mesh.axis2)[i] for i in (ix,iy))
        if ix < iy:
            data = field.array.reshape((len(xaxis), len(yaxis), field.array.shape[-1]))[:,:,[ix,iy]].transpose((1,0,2))
        else:
            data = field.array.reshape((len(yaxis), len(xaxis), field.array.shape[-1]))[:,:,[ix,iy]]
    else:
        raise TypeError("plot_stream can be only used for data obtained for rectangular mesh with regular axes")

    if 'linewidth' in kwargs or 'lw' in kwargs: scale = None

    m0, m1 = meshgrid(array(xaxis), array(yaxis))
    if scale or color == 'norm':
        norm = sum(data**2, 2)
        norm /= norm.max()
    if color == 'norm':
        color = norm

    axes = matplotlib.pylab.gca()
    if ix > iy and not axes.yaxis_inverted():
        axes.invert_yaxis()
    xlabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + ix]))
    ylabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + iy]))

    if scale:
        return streamplot(m0, m1, data[:,:,0].real, data[:,:,1].real, linewidth=scale*norm, color=color, **kwargs)
    else:
        return streamplot(m0, m1, data[:,:,0].real, data[:,:,1].real, color=color, **kwargs)


def plot_boundary(boundary, mesh, geometry, cmap=None, color='0.75', plane=None, zorder=4, **kwargs):
    """Plot points of specified boundary"""
    #TODO documentation

    if not isinstance(mesh, plask.mesh.Mesh):
        raise TypeError("plot_boundary called for non-mesh type")

    if type(cmap) == str: cmap = get_cmap(cmap)
    if cmap is not None: c = []
    else: c = color
    if isinstance(mesh, plask.mesh.Mesh3D):
        ax = _get_2d_axes(plane)
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

    axes = matplotlib.pylab.gca()
    if ax[0] > ax[1] and not axes.yaxis_inverted():
        axes.invert_yaxis()
    xlabel(u"${}$ [µm]".format(plask.config.axes[3 - mesh.dim + ax[0]]))
    ylabel(u"${}$ [µm]".format(plask.config.axes[3 - mesh.dim + ax[1]]))

    return scatter(x, y, c=c, zorder=zorder, cmap=cmap, **kwargs)


# ### plot_mesh ###

def plot_mesh(mesh, color='0.5', lw=1.0, plane=None, set_limits=False, zorder=2):
    """Plot two-dimensional rectilinear mesh."""
    #TODO documentation

    axes = matplotlib.pylab.gca()
    lines = []

    if not isinstance(mesh, plask.mesh.Mesh):
        raise TypeError("plot_mesh called for non-mesh type")

    if isinstance(mesh, plask.mesh.Rectangular2D):
        ix, iy = 0, 1
        y_min = mesh.axis1[0]; y_max = mesh.axis1[-1]
        for x in mesh.axis0:
            lines.append(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=lw, zorder=zorder))
        x_min = mesh.axis0[0]; x_max = mesh.axis0[-1]
        for y in mesh.axis1:
            lines.append(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=lw, zorder=zorder))

    elif isinstance(mesh, plask.mesh.Rectangular3D):
        ix, iy = _get_2d_axes(plane)
        axis = tuple((mesh.axis0, mesh.axis1, mesh.axis2)[i] for i in (ix,iy))

        y_min = axis[1][0]; y_max = axis[1][-1]
        for x in axis[0]:
            lines.append(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=lw, zorder=zorder))
        x_min = axis[0][0]; x_max = axis[0][-1]
        for y in axis[1]:
            lines.append(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=lw, zorder=zorder))

    else:
        raise NotImplementedError("plot_mesh can be only used for data rectangular")

    for line in lines:
        axes.add_line(line)
    if set_limits:
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)

    if ix > iy and not axes.yaxis_inverted():
        axes.invert_yaxis()
    xlabel(u"${}$ [µm]".format(plask.config.axes[3 - mesh.dim + ix]))
    ylabel(u"${}$ [µm]".format(plask.config.axes[3 - mesh.dim + iy]))


    return lines


from ._plot_geometry import plot_geometry
