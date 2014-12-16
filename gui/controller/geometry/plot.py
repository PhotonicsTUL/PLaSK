# -*- coding: utf-8 -*-
# ### plot_geometry ###

import plask
import matplotlib
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.artist

_geometry_drawers = {}

class DrawEnviroment(object):

    def __init__(self, axes, color = 'k', lw = 1.0, z_order=3.0):
        super(DrawEnviroment, self).__init__()
        self.patches = []
        self.color = color
        self.lw = lw
        self.axes = axes
        self.z_order = z_order

    def append(self, artist, clip_box):
        if clip_box is not None: artist.set_clip_box(clip_box)
        self.patches.append(artist)


def _draw_Block(env, geometry_object, transform, clip_box):
    bbox = geometry_object.bbox
    block = matplotlib.patches.Rectangle(
        (bbox.lower[env.axes[0]], bbox.lower[env.axes[1]]),
        bbox.upper[env.axes[0]]-bbox.lower[env.axes[0]], bbox.upper[env.axes[1]]-bbox.lower[env.axes[1]],
        ec=env.color, #color
        lw=env.lw, #line(?)
        fill = False,
        zorder = env.z_order,
        transform = transform
    )
    env.append(block, clip_box)

_geometry_drawers[plask.geometry.Block2D] = _draw_Block
_geometry_drawers[plask.geometry.Block3D] = _draw_Block

def _draw_Triangle(env, geometry_object, transform, clip_box):
    p1 = geometry_object.a
    p2 = geometry_object.b
    env.append(matplotlib.patches.Polygon(
                ((0.0, 0.0), (p1[0], p1[1]), (p2[0], p2[1])),
                closed=True, ec=env.color, lw=env.lw, fill=False, zorder=env.z_order, transform=transform),
               clip_box
    )

_geometry_drawers[plask.geometry.Triangle] = _draw_Triangle

def _draw_Circle(env, geometry_object, transform, clip_box):
    env.append(matplotlib.patches.Circle(
                (0.0, 0.0), geometry_object.radius,
                ec=env.color, lw=env.lw, fill=False, zorder=env.z_order, transform=transform),
               clip_box
    )

_geometry_drawers[plask.geometry.Circle] = _draw_Circle
_geometry_drawers[plask.geometry.Sphere] = _draw_Circle

def _draw_Cylinder(env, geometry_object, transform, clip_box):
    if env.axes != (0, 1) and env.axes != (1, 0):
        _draw_Block(env, geometry_object, transform, clip_box)
    else:
        _draw_Circle(env, geometry_object, transform, clip_box)

_geometry_drawers[plask.geometry.Cylinder] = _draw_Cylinder


def _draw_Translation(env, geometry_object, transform, clip_box):
    new_transform = matplotlib.transforms.Affine2D()
    t = geometry_object.translation
    new_transform.translate(t[env.axes[0]], t[env.axes[1]])
    _draw_geometry_object(env, geometry_object.item, new_transform + transform, clip_box)

_geometry_drawers[plask.geometry.Translation2D] = _draw_Translation
_geometry_drawers[plask.geometry.Translation3D] = _draw_Translation

def _draw_Flip(env, geometry_object, transform, clip_box):
    if geometry_object.axis == 0:
        _draw_geometry_object(env, geometry_object.item, matplotlib.transforms.Affine2D.from_values(-1.0, 0, 0, 1.0, 0, 0) + transform, clip_box)
    else:
        _draw_geometry_object(env, geometry_object.item, matplotlib.transforms.Affine2D.from_values(1.0, 0, 0, -1.0, 0, 0) + transform, clip_box)

_geometry_drawers[plask.geometry.Flip2D] = _draw_Flip

def _draw_Mirror(env, geometry_object, transform, clip_box):
    _draw_geometry_object(env, geometry_object.item, transform, clip_box)
    _draw_Flip(env, geometry_object, transform, clip_box)

_geometry_drawers[plask.geometry.Mirror2D] = _draw_Mirror

#TODO: mirror, flip, clip

def _draw_geometry_object(env, geometry_object, transform, clip_box):
    if geometry_object is None: return
    drawer = _geometry_drawers.get(type(geometry_object))
    if drawer is None:
        try:
            for child in geometry_object:
                _draw_geometry_object(env, child, transform, clip_box)
        except TypeError:
            pass    #ignore non-iterable object
    else:
        drawer(env, geometry_object, transform, clip_box)

def plot_geometry_object(figure, geometry, color='k', lw=1.0, plane=None, set_limits=False, zorder=3, mirror=False):
    '''Plot geometry.'''
    #TODO documentation

    #figure.clear()
    axes = figure.add_subplot(111)

    if type(geometry) == plask.geometry.Cartesian3D:
        dd = 0
        #if plane is None: plane = 'xy'
        ax = _get_2d_axes(plane)
        dirs = tuple((("back", "front"), ("left", "right"), ("top", "bottom"))[i] for i in ax)
    else:
        dd = 1
        ax = (0,1)
        dirs = (("inner", "outer") if type(geometry) == plask.geometry.Cylindrical2D else ("left", "right"),
                ("top", "bottom"))

    env = DrawEnviroment(ax, color, lw, z_order=zorder)

    hmirror = mirror and (geometry.borders[dirs[0][0]] == 'mirror' or geometry.borders[dirs[0][1]] == 'mirror' or dirs[0][0] == "inner")
    vmirror = mirror and (geometry.borders[dirs[1][0]] == 'mirror' or geometry.borders[dirs[1][1]] == 'mirror')

    _draw_geometry_object(env, geometry, axes.transData, None)
    if vmirror:
        _draw_geometry_object(env, geometry, matplotlib.transforms.Affine2D.from_values(-1.0, 0, 0, 1.0, 0, 0) + axes.transData, None)
    if hmirror:
        _draw_geometry_object(env, geometry, matplotlib.transforms.Affine2D.from_values(1.0, 0, 0, -1.0, 0, 0) + axes.transData, None)
    #for trans,box in zip(geometry.get_leafs_translations(), geometry.get_leafs_bboxes()):
    #    if box:
    #        _geometry_plotters[type(trans.item)](patches, trans, box, ax, hmirror, vmirror, color, lw, zorder)

    for patch in env.patches:
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

    #TODO changed 9 XII 2014 in order to avoid segfault in GUI:
    # matplotlib.pyplot.xlabel -> axes.set_xlabel
    # matplotlib.pyplot.ylabel -> axes.set_ylabel
    axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[0]]))
    axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[1]]))

    return env.patches






# ------------ old plot code: -------------------

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

    #TODO changed 9 XII 2014 in order to avoid segfault in GUI:
    # matplotlib.pyplot.xlabel -> axes.set_xlabel
    # matplotlib.pyplot.ylabel -> axes.set_ylabel
    axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[0]]))
    axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[1]]))

    return patches