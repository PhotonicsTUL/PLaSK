# -*- coding: utf-8 -*-
"""
This is an interface to Matplotlib and pylab.

It contains some helper functions helping with plotting PLaSK
input/output.

Additional functions defined here are:

TODO

"""

import sys as _sys
import os as _os

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
        for n in ("QString", "QVariant"):
            try:
                sip.setapi(n, 2)
            except:
                pass
    except:
        pass
# Fix for Anaconda bug
elif backend == 'Qt5Agg' and _os.name == 'nt':
    from PyQt5 import QtWidgets as _QtWidgets
    _QtWidgets.QApplication.addLibraryPath(_os.path.join(_sys.prefix, 'Library', 'plugins'))
    _QtWidgets.QApplication.addLibraryPath(_os.path.join(_os.path.dirname(_QtWidgets.__file__), 'plugins'))

import matplotlib.pylab
from matplotlib.pylab import *
# __doc__ += matplotlib.pylab.__doc__


def aspect(aspect, adjustable=None, anchor=None):
    gca().set_aspect(aspect, adjustable, anchor)
aspect.__doc__ = Axes.set_aspect.__doc__


def window_title(title):
    """
    Set the title text of the window containing the figure.  Note that
    this has no effect if there is no window (e.g., a PS backend).
    """
    gcf().canvas.set_window_title(title)

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
            if len(comp) == 1: comp = comp + comp # x - > xx
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


def plot_field(field, levels=16, plane=None, fill=True, antialiased=False, comp=None, axes=None, figure=None, **kwargs):
    """
    Plot scalar real fields as two-dimensional color map.

    This function uses ``contourf`` of ``contour`` functions to plot scalar real
    fields returned by providers. It can also plot a single component of a vector
    or tensor field; in such case the component must be specified with the ``comp``
    argument.

    Args:
        field (Data): The field to plot. As it is usually returned by providers, it
                      already contains the mesh and field values.

        levels (int or sequence): Number of value bands to plot of a sequence
                                  containing the bands.

        plane (str): If the field to plot is a 3D one, this argument must be used
                     to select to which the field is projected. The field mesh must
                     be flat in this plane i.e. all its points must lie at the same
                     level alongside the axis perpendicular to the specified plane.

        fill (bool): If True, ``contourf`` is used to plot the field i.e. the bands
                     are filled. Otherwise the coutours are plotted with
                     ``countour``.

        antialiased (bool): If True, the antialiasing is enabled.

        comp (int or str): If the vector field is plotted, this argument must
                           specify the component to plot. It can be either
                           a component number or its name.

        axes (Axes): Matplotlib axes to which the geometry is drawn. If *None*
                (the default), new axes are created.

        figure (Figure): Matplotlib destination figure. This parameter is
                ignored if `axes` are given. In *None*, the geometry
                is plotted to the current figure.

        **kwargs: Keyword arguments passed to ``contourf`` or ``contour``.
    """

    if axes is None:
        if figure is None:
            axes = gca()
        else:
            axes = figure.add_subplot(111)

    if isinstance(field.mesh, plask.mesh.Rectangular2D):
        if fill and levels is None:
            xaxis = plask.mesh.Regular(field.mesh.axis0[0], field.mesh.axis0[-1], len(field.mesh.axis0))
            yaxis = plask.mesh.Regular(field.mesh.axis1[0], field.mesh.axis1[-1], len(field.mesh.axis1))
            field = field.interpolate(plask.mesh.Rectangular2D(xaxis, yaxis), 'linear')
        else:
            xaxis = field.mesh.axis0
            yaxis = field.mesh.axis1
        data = field.array.real
        ax = 0, 1
        if len(data.shape) == 3:
            if comp is None:
                raise TypeError("Specify {} component to plot".format('tensor' if field.dtype == tuple else 'vector'))
            else:
                comp = _get_component(comp, data.shape[2])
                data = data[:,:,comp]
        data = data.transpose()
    elif isinstance(field.mesh, plask.mesh.Rectangular3D):
        if fill and levels is None:
            axis0 = plask.mesh.Regular(field.mesh.axis0[0], field.mesh.axis0[-1], len(field.mesh.axis0))
            axis1 = plask.mesh.Regular(field.mesh.axis1[0], field.mesh.axis1[-1], len(field.mesh.axis1))
            axis2 = plask.mesh.Regular(field.mesh.axis2[0], field.mesh.axis2[-1], len(field.mesh.axis2))
            field = field.interpolate(plask.mesh.Rectangular3D(axis0, axis1, axis2), 'linear')
        data = field.array.real
        if plane is None:
            if data.shape[2] == 1: ax = 1, 0
            elif data.shape[1] == 1: ax = 0, 2
            elif data.shape[0] == 1: ax = 1, 2
            else: raise ValueError("Field mesh must have one dimension equal to 1")
        else:
            ax = _get_2d_axes(plane)
            if data.shape[3-sum(ax)] != 1:
                raise ValueError("Field mesh must have dimension {} equal to 1".format(3-sum(ax)))
        xaxis, yaxis = ((field.mesh.axis0, field.mesh.axis1, field.mesh.axis2)[i] for i in ax)
        if len(data.shape) == 4:
            if comp is None:
                raise TypeError("Specify {} component to plot".format('tensor' if field.dtype == tuple else 'vector'))
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
        if levels is not None:
            result = axes.contourf(xaxis, yaxis, data, levels, antialiased=antialiased, **kwargs)
        else:
            result = axes.imshow(data, extent=(xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]), origin='lower', **kwargs)
    else:
        if 'colors' not in kwargs and 'cmap' not in kwargs:
            result = axes.contour(xaxis, yaxis, data, levels, colors='k', antialiased=antialiased, **kwargs)
        else:
            result = axes.contour(xaxis, yaxis, data, levels, antialiased=antialiased, **kwargs)

    if ax[0] > ax[1] and not axes.yaxis_inverted():
        axes.invert_yaxis()
    axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + ax[0]]))
    axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + ax[1]]))

    if axes == gca(): sci(result)
    return result


def plot_profile(field, comp=None, swap_axes=False, axes=None, figure=None, **kwargs):
    """
    Plot a scalar real field value along one axis.

    This function creates a classical plot of a scalar field. The field must be
    obtained on a rectangular mesh that has a single point in all dimensions but
    one. In other words, the field must be obtained over a single line which
    is used as an argument axis of this plot.

    Args:
        field (Data): The field to plot. As it is usually returned by providers, it
                      already contains the mesh and field values.

        comp (int or str): If the vector field is plotted, this argument must
                           specify the component to plot. It can be either
                           a component number or its name.

        swap_axes (bool): If False, the mesh position is plotted on the horizontal
                          axis and the field value on the vertical one and otherwise
                          if this argument is True.

        axes (Axes): Matplotlib axes to which the geometry is drawn. If *None*
                (the default), new axes are created.

        figure (Figure): Matplotlib destination figure. This parameter is
                ignored if `axes` are given. In *None*, the geometry
                is plotted to the current figure.

        **kwargs: Keyword arguments passed to ``plot`` function.
    """

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

    if axes is None:
        if figure is None:
            axes = gca()
        else:
            axes = figure.add_subplot(111)

    if swap_axes:
        axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[ax]))
        return axes.plot(data.ravel(), axis, **kwargs)
    else:
        axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[ax]))
        return axes.plot(axis, data.ravel(), **kwargs)


def plot_vectors(field, plane=None, axes=None, figure=None, angles='xy', scale_units='xy', **kwargs):
    """
    Plot vector field with arrows.

    This function uses ``quiver`` to plot a vector field returned by some providers
    with arrows.

    Args:
        field (Data): The field to plot. As it is usually returned by providers, it
                      already contains the mesh and field values.

        plane (str): If the field to plot is a 3D one, this argument must be used
                     to select to which the field is projected. The field mesh must
                     be flat in this plane i.e. all its poinst must lie at the same
                     level alongside the axis perpendicular to the specified plane.

        axes (Axes): Matplotlib axes to which the geometry is drawn. If *None*
                (the default), new axes are created.

        figure (Figure): Matplotlib destination figure. This parameter is
                ignored if `axes` are given. In *None*, the geometry
                is plotted to the current figure.

        angles (str): This is equivalent to the ``angles`` argument of ``quiver``,
                      however, the default value is 'xy', which makes more sense
                      for the physical fields.

        scale_units (str): ``scale_units`` argument of ``quiver``, with 'xy' as
                           the default.

        **kwargs: Keyword arguments passed to ``quiver``.
    """

    if axes is None:
        if figure is None:
            axes = gca()
        else:
            axes = figure.add_subplot(111)

    m = field.mesh

    if isinstance(m, plask.mesh.Rectangular2D):
        ix, iy = 0, 1
        xaxis, yaxis = m.axis0, m.axis1
        data = field.array.transpose((1,0,2))
    elif isinstance(m, plask.mesh.Rectangular3D):
        ix, iy = _get_2d_axes(plane)
        if field.array.shape[3-ix-iy] != 1:
            raise ValueError("Field mesh must have dimension {} equal to 1".format(3-ix-iy))
        xaxis, yaxis = ((field.mesh.axis0, field.mesh.axis1, field.mesh.axis2)[i] for i in (ix,iy))
        if ix < iy:
            data = field.array.reshape((len(xaxis), len(yaxis), field.array.shape[-1]))[:,:,[ix,iy]].transpose((1,0,2))
        else:
            data = field.array.reshape((len(yaxis), len(xaxis), field.array.shape[-1]))[:,:,[ix,iy]]
    else:
        raise NotImplementedError("mesh type not supported")

    if ix > iy and not axes.yaxis_inverted():
        axes.invert_yaxis()
    axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + ix]))
    axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + iy]))

    result = axes.quiver(array(xaxis), array(yaxis), data[:,:,0].real, data[:,:,1].real,
                         angles=angles, scale_units=scale_units, **kwargs)
    if axes == gca(): sci(result)
    return result


def plot_stream(field, plane=None, axes=None, figure=None, scale=8.0, color='k', **kwargs):
    """
    Plot vector field as a streamlines.

    This function uses ``streamplot`` to plot a vector field returned by some
    providers using streamlines.

    Args:
        field (Data): The field to plot. As it is usually returned by providers, it
                      already contains the mesh and field values.

        plane (str): If the field to plot is a 3D one, this argument must be used
                     to select to which the field is projected. The field mesh must
                     be flat in this plane i.e. all its poinst must lie at the same
                     level alongside the axis perpendicular to the specified plane.

        axes (Axes): Matplotlib axes to which the geometry is drawn. If *None*
                (the default), new axes are created.

        figure (Figure): Matplotlib destination figure. This parameter is
                ignored if `axes` are given. In *None*, the geometry
                is plotted to the current figure.

        scale (float): Scale by which the streamlines widths are multiplied.

        color (str): Color of the streamlines.

        **kwargs: Keyword arguments passed to ``streamplot``.
    """

    if axes is None:
        if figure is None:
            axes = gca()
        else:
            axes = figure.add_subplot(111)

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

    if ix > iy and not axes.yaxis_inverted():
        axes.invert_yaxis()
    axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + ix]))
    axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[3 - field.mesh.dim + iy]))

    if scale:
        result = axes.streamplot(m0, m1, data[:,:,0].real, data[:,:,1].real, linewidth=scale*norm, color=color, **kwargs)
    else:
        result = axes.streamplot(m0, m1, data[:,:,0].real, data[:,:,1].real, color=color, **kwargs)

    if axes == gca(): sci(result.lines)
    return result


def plot_boundary(boundary, mesh, geometry, colors=None, color='0.75', plane=None, axes=None, figure=None, zorder=4, **kwargs):
    """
    Plot boundary conditions.

    This functions is used to visualize boundary conditions. It plots the markers at
    mesh points, in which boundary conditions are specified. Optionally it can color
    the points according to the boundary condition value using a specified colormap.

    Args:
        boundary (BoundaryConditions): Boundary conditions to plot. Normally, this
            is some attribute of a solver.

        mesh (plask.mesh.Mesh): Mesh which points are selected as boundary
            conditions. Normally it should be the mesh configured for the solver
            whose boundary conditions are plotted.

        geometry (plask.geometry.Geometry): Geometry over, which the boundary
            conditions are defined. Normally it should be the geometry configured
            for the solver whose boundary conditions are plotted.

        colors (str): Sequence of colors of the boundary conditions points. The
            length ot this sequence must be equal to the number of distinct boundary
            conditions. If this is None, all points have the same color.

        color (str): Color of the boundary conditions points if ``colors`` is None.

        plane (str): If the field to plot is a 3D one, this argument must be used
            to select to which the field is projected. The field mesh must be flat
            in this plane i.e. all its poinst must lie at the same level alongside
            the axis perpendicular to the specified plane.

        axes (Axes): Matplotlib axes to which the geometry is drawn. If *None*
                (the default), new axes are created.

        figure (Figure): Matplotlib destination figure. This parameter is
                ignored if `axes` are given. In *None*, the geometry
                is plotted to the current figure.

        zorder (float): Ordering index of the geometry plot in the graph.
                Elements with higher `zorder` are drawn on top of the ones
                with the lower one.

        **kwargs: Keyword arguments passed to ``scatter``.

    Example:
        >>> solver = electrical.Schockey2D()
        >>> # configure solver
        >>> plot_boundary(solver.voltage_boundary, solver.mesh, solver.geometry,
        ...               cmap='summer')
    """

    if axes is None:
        if figure is None:
            axes = gca()
        else:
            axes = figure.add_subplot(111)

    if not isinstance(mesh, plask.mesh.Mesh):
        if isinstance(mesh, (plask.mesh.Generator1D, plask.mesh.Generator2D, plask.mesh.Generator3D)):
            mesh = mesh(geometry)
        else:
            raise TypeError("plot_boundary called for non-mesh type")

    if isinstance(mesh, plask.mesh.Mesh3D):
        ax = _get_2d_axes(plane)
    else:
        ax = (0,1)
    if ax[0] > ax[1] and not axes.yaxis_inverted():
        axes.invert_yaxis()

    scatters = []
    for i, (place, value) in enumerate(boundary):
        x = []
        y = []
        points = place(mesh, geometry)
        for j in points:
            x.append(mesh[j][ax[0]])
            y.append(mesh[j][ax[1]])
        if colors is not None:
            color = colors[i]
        scatters.append(axes.scatter(x, y, c=color, zorder=zorder, **kwargs))

    xlabel(u"${}$ [µm]".format(plask.config.axes[3 - mesh.dim + ax[0]]))
    ylabel(u"${}$ [µm]".format(plask.config.axes[3 - mesh.dim + ax[1]]))

    return scatters


# ### plot_mesh ###

def plot_mesh(mesh, color='0.5', lw=1.0, plane=None, margin=False, axes=None, figure=None, zorder=1.5, alpha=1.0):
    """
    Plot two-dimensional mesh.

    Args:
        mesh (plask.mesh.Mesh): Mesh to draw.

        color (str): Color of the drawn mesh lines.

        lw (float): Width of the drawn mesh lines.

        plane (str): Planes to draw. Should be a string with two letters
                specifying axis names of the drawn plane. This argument
                is required if 3D mesh is plotted and ignored for 2D meshes.

        margin (float of None): The margins around the structure (as a fraction
                of the structure bounding box) to which the plot limits should
                be set. If None, the axes limits are not adjusted.

        axes (Axes): Matplotlib axes to which the geometry is drawn. If *None*
                (the default), new axes are created.

        figure (Figure): Matplotlib destination figure. This parameter is
                ignored if `axes` are given. In *None*, the geometry
                is plotted to the current figure.

        zorder (float): Ordering index of the meshs plot in the graph.
                Elements with higher `zorder` are drawn on top of the ones
                with the lower one.

        alpha (float): Opacity of the drawn mesh (1: fully opaque,
                0: fully transparent)
    """

    if axes is None:
        if figure is None:
            axes = gca()
        else:
            axes = figure.add_subplot(111)

    lines = []

    if not isinstance(mesh, plask.mesh.Mesh):
        raise TypeError("plot_mesh called for non-mesh type")

    if isinstance(mesh, plask.mesh.Rectangular2D):
        ix, iy = 0, 1
        y_min = mesh.axis1[0]; y_max = mesh.axis1[-1]
        for x in mesh.axis0:
            lines.append(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=lw, zorder=zorder, alpha=alpha))
        x_min = mesh.axis0[0]; x_max = mesh.axis0[-1]
        for y in mesh.axis1:
            lines.append(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=lw, zorder=zorder, alpha=alpha))

    elif isinstance(mesh, plask.mesh.Rectangular3D):
        ix, iy = _get_2d_axes(plane)
        axis = tuple((mesh.axis0, mesh.axis1, mesh.axis2)[i] for i in (ix,iy))

        y_min = axis[1][0]; y_max = axis[1][-1]
        for x in axis[0]:
            lines.append(matplotlib.lines.Line2D([x,x], [y_min,y_max], color=color, lw=lw, zorder=zorder, alpha=alpha))
        x_min = axis[0][0]; x_max = axis[0][-1]
        for y in axis[1]:
            lines.append(matplotlib.lines.Line2D([x_min,x_max], [y,y], color=color, lw=lw, zorder=zorder, alpha=alpha))

    elif isinstance(mesh, (plask.mesh.Regular, plask.mesh.Ordered)):
        ix, iy = 0, 1
        for x in mesh:
            lines.append(axes.axvline(x, color=color, lw=lw, zorder=zorder, alpha=alpha))
        x_min = mesh[0]; x_max = mesh[-1]
        y_max = y_min = None

    else:
        raise NotImplementedError("plot_mesh can be only used for rectangular mesh")

    for line in lines:
        axes.add_line(line)
    if margin:
        m0 = (x_max - x_min) * margin
        axes.set_xlim(x_min - m0, x_max + m0)
        if y_max is not None:
            m1 = (y_max - y_min) * margin
            axes.set_ylim(y_min - m1, y_max + m1)

    if ix > iy and not axes.yaxis_inverted():
        axes.invert_yaxis()
    dim = max(2, mesh.dim)
    xlabel(u"${}$ [µm]".format(plask.config.axes[3 - dim + ix]))
    ylabel(u"${}$ [µm]".format(plask.config.axes[3 - dim + iy]))


    return lines


from ._plot_geometry import plot_geometry, DEFAULT_COLORS, ColorFromDict
