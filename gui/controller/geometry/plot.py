# -*- coding: utf-8 -*-
# ### plot_geometry ###

import plask
import matplotlib
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.artist

_geometry_plotters = {}

def _add_path_Block(patches, trans, box, ax, hmirror, vmirror, color, lw, zorder):
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
                                                        ec=color, lw=lw, fill=False, zorder=zorder))
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

def _add_path_Triangle(patches, trans, box, ax, hmirror, vmirror, color, lw, zorder):
    p0 = trans.translation
    p1 = p0 + trans.item.a
    p2 = p0 + trans.item.b
    patches.append(matplotlib.patches.Polygon(((p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1])),
                                              closed=True, ec=color, lw=lw, fill=False, zorder=zorder))
    if vmirror:
        patches.append(matplotlib.patches.Polygon(((p0[0], -p0[1]), (p1[0], -p1[1]), (p2[0], p2[1])),
                                                  closed=True, ec=color, lw=lw, fill=False, zorder=zorder))
    if hmirror:
        patches.append(matplotlib.patches.Polygon(((-p0[0], p0[1]), (-p1[0], p1[1]), (-p2[0], p2[1])),
                                                  closed=True, ec=color, lw=lw, fill=False, zorder=zorder))
        if vmirror:
            patches.append(matplotlib.patches.Polygon(((-p0[0], -p0[1]), (-p1[0], -p1[1]), (-p2[0], -p2[1])),
                                                      closed=True, ec=color, lw=lw, fill=False, zorder=zorder))

_geometry_plotters[plask.geometry.Triangle] = _add_path_Triangle

def _add_path_Circle(patches, trans, box, ax, hmirror, vmirror, color, lw, zorder):
    tr = trans.translation
    vecs = [ tr ]
    if hmirror: vecs.append(plask.vec(-tr[0], tr[1]))
    if vmirror: vecs.append(plask.vec(tr[0], -tr[1]))
    if hmirror and vmirror: vecs.append(plask.vec(-tr[0], -tr[1]))
    for vec in vecs:
        patches.append(matplotlib.patches.Circle(vec, trans.item.radius, ec=color, lw=lw, fill=False, zorder=zorder))

_geometry_plotters[plask.geometry.Circle] = _add_path_Circle

def _add_path_Cylinder(patches, trans, box, ax, hmirror, vmirror, color, lw, zorder):
    if ax != (0,1) and ax != (1,0):
        _add_path_Block(patches, trans, box, ax, hmirror, vmirror, color, lw, zorder)
    else:
        tr = trans.translation
        if ax == (1,0): tr = plask.vec(tr[1], tr[0])
        vecs = [ tr ]
        if hmirror: vecs.append(plask.vec(-tr[0], tr[1]))
        if vmirror: vecs.append(plask.vec(tr[0], -tr[1]))
        if hmirror and vmirror: vecs.append(plask.vec(-tr[0], -tr[1]))
        for vec in vecs:
            patches.append(matplotlib.patches.Circle(vec, trans.item.radius,
                                                     ec=color, lw=lw, fill=False, zorder=zorder))
_geometry_plotters[plask.geometry.Cylinder] = _add_path_Cylinder

def _add_path_Sphere(patches, trans, box, ax, hmirror, vmirror, color, lw, zorder):
    tr = trans.translation[ax[0]], trans.translation[ax[1]]
    vecs = [ tr ]
    if hmirror: vecs.append(plask.vec(-tr[0], tr[1]))
    if vmirror: vecs.append(plask.vec(tr[0], -tr[1]))
    if hmirror and vmirror: vecs.append(plask.vec(-tr[0], -tr[1]))
    for vec in vecs:
        patches.append(matplotlib.patches.Circle(vec, trans.item.radius, ec=color, lw=lw, fill=False, zorder=zorder))

_geometry_plotters[plask.geometry.Sphere] = _add_path_Sphere


def plot_geometry(figure, geometry, color='k', lw=1.0, plane=None, set_limits=False, zorder=3, mirror=False):
    '''Plot geometry.'''
    #TODO documentation

    #figure.clear()
    axes = figure.add_subplot(111)
    patches = []

    if type(geometry) == plask.geometry.Cartesian3D:
        dd = 0
        ax = _get_2d_axes(plane)
        dirs = tuple((("back", "front"), ("left", "right"), ("top", "bottom"))[i] for i in ax)
    else:
        dd = 1
        ax = (0,1)
        dirs = (("inner", "outer") if type(geometry) == plask.geometry.Cylindrical2D else ("left", "right"),
                ("top", "bottom"))

    hmirror = mirror and (geometry.borders[dirs[0][0]] == 'mirror' or geometry.borders[dirs[0][1]] == 'mirror' or dirs[0][0] == "inner")
    vmirror = mirror and (geometry.borders[dirs[1][0]] == 'mirror' or geometry.borders[dirs[1][1]] == 'mirror')

    for trans,box in zip(geometry.get_leafs_translations(), geometry.get_leafs_bboxes()):
        if box:
            _geometry_plotters[type(trans.item)](patches, trans, box, ax, hmirror, vmirror, color, lw, zorder)

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

    if ax[0] > ax[1] and not axes.yaxis_inverted():
        axes.invert_yaxis()
    matplotlib.pyplot.xlabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[0]]))
    matplotlib.pyplot.ylabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[1]]))

    return patches