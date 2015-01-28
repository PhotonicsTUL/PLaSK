# -*- coding: utf-8 -*-
# ### plot_geometry ###
import math

import plask
import matplotlib
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.artist

import collections
from zlib import adler32

from pylab import _get_2d_axes

__all__ = ('plot_geometry')


_geometry_drawers = {}


class BBoxIntersection(matplotlib.transforms.BboxBase):
    """
    A :class:`Bbox` equals to intersection of 2 other bounding boxes.
    When any of the children box changes, the bounds of this bbox will update accordingly.
    """
    def __init__(self, bbox1, bbox2, **kwargs):
        """
        :param bbox1: a first child :class:`Bbox`
        :param bbox2: a second child :class:`Bbox`
        """
        assert bbox1.is_bbox
        assert bbox2.is_bbox

        matplotlib.transforms.BboxBase.__init__(self, **kwargs)
        self._bbox1 = bbox1
        self._bbox2 = bbox2
        self.set_children(bbox1, bbox2)

    def __repr__(self):
        return "BBoxIntersection(%r, %r)" % (self._bbox, self._bbox)

    def get_points(self):
        if self._invalid:
            box = matplotlib.transforms.Bbox.intersection(self._bbox1, self._bbox2)
            if box is not None:
                self._points = box.get_points()
            else:
                self._points = matplotlib.transforms.Bbox.from_bounds(0, 0, -1, -1).get_points()
            self._invalid = 0
        return self._points
    get_points.__doc__ = matplotlib.transforms.Bbox.get_points.__doc__


def material_to_color(material):
    """
        Generate color for given material.
        :param plask.Material material: material
        :return (float, float, float): RGB color, 3 floats, each in range [0, 1]
    """
    i = adler32(str(material))      #maybe crc32?
    return (i & 0xff) / 255.0, ((i >> 8) & 0xff) / 255.0, ((i >> 16) & 0xff) / 255.0


class DrawEnviroment(object):
    """
        Drawing configuration.
    """

    def __init__(self, axes, artist_dst, fill = False, color = 'k', lw = 1.0, zorder=3.0):
        """
        :param axes: plane to draw (important in 3D)
        :param artist_dst: mpl axis where artist should be appended
        :param bool fill: True if artists should be filled
        :param color: edge color (mpl format)
        :param lw: line width
        :param zorder: artists z order
        """
        super(DrawEnviroment, self).__init__()
        self.artist_dst = artist_dst
        self.fill = fill
        self.color = color
        self.lw = lw
        self.axes = axes
        self.zorder = zorder

    def append(self, artist, clip_box, geometry_object):
        """
        Configure and append artist to destination axis object.
        :param artist: artist to append
        :param matplotlib.transforms.BboxBase clip_box: clipping box for artist, optional
        :param geometry_object: plask's geometry object which is represented by the artist
        """
        if self.fill:
            artist.set_fill(True)
            artist.set_facecolor(material_to_color(geometry_object.representative_material))
        else:
            artist.set_fill(False)
        artist.set_linewidth(self.lw)
        artist.set_ec(self.color)
        self.artist_dst.add_patch(artist)
        if clip_box is not None:
            artist.set_clip_box(BBoxIntersection(clip_box, artist.get_clip_box()))
            #artist.set_clip_box(clip_box)
            #artist.set_clip_on(True)
            #artist.set_clip_path(clip_box)
        artist.set_zorder(self.zorder)


def _draw_Block(env, geometry_object, transform, clip_box):
    bbox = geometry_object.bbox
    block = matplotlib.patches.Rectangle(
        (bbox.lower[env.axes[0]], bbox.lower[env.axes[1]]),
        bbox.upper[env.axes[0]]-bbox.lower[env.axes[0]], bbox.upper[env.axes[1]]-bbox.lower[env.axes[1]],
        transform=transform
    )
    env.append(block, clip_box, geometry_object)

_geometry_drawers[plask.geometry.Block2D] = _draw_Block
_geometry_drawers[plask.geometry.Block3D] = _draw_Block


def _draw_Triangle(env, geometry_object, transform, clip_box):
    p1 = geometry_object.a
    p2 = geometry_object.b
    env.append(matplotlib.patches.Polygon(((0.0, 0.0), (p1[0], p1[1]), (p2[0], p2[1])), closed=True, transform=transform),
               clip_box, geometry_object
    )


_geometry_drawers[plask.geometry.Triangle] = _draw_Triangle


def _draw_Circle(env, geometry_object, transform, clip_box):
    env.append(matplotlib.patches.Circle((0.0, 0.0), geometry_object.radius, transform=transform),
               clip_box, geometry_object
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
    if geometry_object.axis_nr == env.axes[0]:
        _draw_geometry_object(env, geometry_object.item, matplotlib.transforms.Affine2D.from_values(-1.0, 0, 0, 1.0, 0, 0) + transform, clip_box)
    elif geometry_object.axis_nr == env.axes[1]:
        _draw_geometry_object(env, geometry_object.item, matplotlib.transforms.Affine2D.from_values(1.0, 0, 0, -1.0, 0, 0) + transform, clip_box)
    else:
        _draw_geometry_object(env, geometry_object.item, transform, clip_box)

_geometry_drawers[plask.geometry.Flip2D] = _draw_Flip
_geometry_drawers[plask.geometry.Flip3D] = _draw_Flip

def _draw_Mirror(env, geometry_object, transform, clip_box):
    _draw_geometry_object(env, geometry_object.item, transform, clip_box)
    if geometry_object.axis_nr in env.axes: # in 3D this must not be true
        _draw_Flip(env, geometry_object, transform, clip_box)

_geometry_drawers[plask.geometry.Mirror2D] = _draw_Mirror
_geometry_drawers[plask.geometry.Mirror3D] = _draw_Mirror

def _draw_clipped(env, geometry_object, transform, clip_box, new_clip_box):
    """Used by _draw_Clip and _draw_Intersection."""
    def _b(bound):
        return math.copysign(1e100, bound) if math.isinf(bound) else bound

    new_clipbox = matplotlib.transforms.TransformedBbox(
       #matplotlib.transforms.Bbox([
       #    [obj_box.lower[env.axes[0]], obj_box.lower[env.axes[1]]],
       #    [obj_box.upper[env.axes[0]], obj_box.upper[env.axes[1]]]
       #]),
       matplotlib.transforms.Bbox.from_extents(_b(new_clip_box.lower[env.axes[0]]), _b(new_clip_box.lower[env.axes[1]]),
                                               _b(new_clip_box.upper[env.axes[0]]), _b(new_clip_box.upper[env.axes[1]])),
       transform
    )

    if clip_box is None:
        clip_box = new_clipbox
    else:
        clip_box = BBoxIntersection(clip_box, new_clipbox)

    x0, y0, x1, y1 = clip_box.extents
    if x0 < x1 and y0 < y1:
        _draw_geometry_object(env, geometry_object, transform, clip_box)
    # else, if clip_box is empty now, it will be never non-empty, so all will be clipped-out


def _draw_Clip(env, geometry_object, transform, clip_box):
    _draw_clipped(env, geometry_object.item, transform, clip_box, geometry_object.clip_box)

_geometry_drawers[plask.geometry.Clip2D] = _draw_Clip
_geometry_drawers[plask.geometry.Clip3D] = _draw_Clip

def _draw_Intersection(env, geometry_object, transform, clip_box):
    _draw_clipped(env, geometry_object.item, transform, clip_box, geometry_object.envelope.bbox)

_geometry_drawers[plask.geometry.Intersection2D] = _draw_Intersection
_geometry_drawers[plask.geometry.Intersection3D] = _draw_Intersection


def _draw_geometry_object(env, geometry_object, transform, clip_box):
    """
    Draw geometry object.
    :param DrawEnviroment env: drawing configuration
    :param geometry_object: object to draw
    :param transform: transform from a plot coordinates to the geometry_object
    :param matplotlib.transforms.BboxBase clip_box: clipping box in plot coordinates
    """
    if geometry_object is None: return
    drawer = _geometry_drawers.get(type(geometry_object))
    if drawer is None:
        try:
            for child_index in range(0, geometry_object._children_len()):
                _draw_geometry_object(env, geometry_object._child(child_index), transform, clip_box)
            #for child in geometry_object:
            #    _draw_geometry_object(env, child, transform, clip_box)
        except TypeError:
            pass    #ignore non-iterable object
    else:
        drawer(env, geometry_object, transform, clip_box)


class ColorFromDict(object):
    """Get color from dict: material name string -> color or using material_to_color (for names which are not present in dict)."""

    def __init__(self, material_dict):
        super(ColorFromDict, self).__init__()
        self.material_dict = dict

    def __call__(self, material):
        try:
            return self.material_dict[str(material)]
        except KeyError:
            return material_to_color(material)


def plot_geometry(geometry, color='k', lw=1.0, plane=None, zorder=2.0, mirror=False, fill=False,
                  axes=None, figure=None, margin=None, get_color=material_to_color, set_limits=None):
    """
    Plot specified geometry.
    
    Args:
        geometry (plask.Geometry): Geometry to draw.

        color (str): Color of the edges of drawn elements.

        lw (float): Width of the edges of drawn elements.

        plane (str): Planes to draw. Should be a string with two letters
                specifying axis names of the drawn plane. This argument
                is required if 3D geometry is plotted and ignored for
                2D geometries.

        zorder (float): Ordering index of the geometry plot in the graph.
                Elements with higher `zorder` are drawn on top of the ones
                with the lower one.

        mirror (bool): If *True* then the geometry is mirrored if its
                specification says so (i.e. some borders are set to
                *mirror* of the geometry is a cylindrical one).

        fill (bool): If True, drawn geometry objects will be filled with colors
                that depends on their material. For Cartesian3D geometry this
                is not supported and then the `fill` parameter is ignored.

        axes (Axes): Matplotlib axes to which the geometry is drawn. If *None*
                (the default), new axes are created.

        figure (Figure): Matplotlib destination figure. This parameter is
                ignored if `axes` are given. In *None*, the geometry
                is plotted to the current figure.

        margin (float of None): The margins around the structure (as a fraction
                of the structure bounding box) to which the plot limits should
                be set. If None, the axes limits are not adjusted.

        get_color (callable): Callable that gets color for material given as
                its parameter or dictionary from material names (strings)
                to colors. Material color should be given as a triple
                (float, float, float) of red, green, blue components, each in
                range [0, 1]. Any other format accepted by set_facecolor()
                method of matplotlib Artist should work as well.

    Returns:
        matplotlib.axes.Axes: appended or given axes object

    Limitations:
        Intersection is not drawn precisely (item is clipped to bonding box of
        the envelope).

        Filling is not supported when Cartesian3D geometry is drawn.
    """

    if set_limits is not None:
        plask.print_log('warning', "plot_geometry: 'set_limits' is obsolette, set 'margin' instead")
        if margin is None:
            margin = 0.

    if not isinstance(get_color, collections.Callable):
        get_color = ColorFromDict(get_color)

    if axes is None:
        if figure is None:
            axes = matplotlib.pylab.gca()
        else:
            axes = figure.add_subplot(111)

    if isinstance(geometry, plask.geometry.Cartesian3D):
        fill = False    # we ignore fill parameter in 3D
        dd = 0
        #if plane is None: plane = 'xy'
        ax = _get_2d_axes(plane)
        dirs = tuple((("back", "front"), ("left", "right"), ("top", "bottom"))[i] for i in ax)
    else:
        dd = 1
        ax = (0,1)
        dirs = (("inner", "outer") if type(geometry) == plask.geometry.Cylindrical2D else ("left", "right"),
                ("top", "bottom"))

    env = DrawEnviroment(ax, axes, fill, color, lw, zorder=zorder)

    hmirror = mirror and (geometry.borders[dirs[0][0]] == 'mirror' or geometry.borders[dirs[0][1]] == 'mirror' or dirs[0][0] == "inner")
    vmirror = mirror and (geometry.borders[dirs[1][0]] == 'mirror' or geometry.borders[dirs[1][1]] == 'mirror')

    _draw_geometry_object(env, geometry, axes.transData, None)
    if vmirror:
        _draw_geometry_object(env, geometry, matplotlib.transforms.Affine2D.from_values(-1.0, 0, 0, 1.0, 0, 0) + axes.transData, None)
    if hmirror:
        _draw_geometry_object(env, geometry, matplotlib.transforms.Affine2D.from_values(1.0, 0, 0, -1.0, 0, 0) + axes.transData, None)

    if margin is not None:
        box = geometry.bbox
        if hmirror:
            m = max(abs(box.lower[ax[0]]), abs(box.upper[ax[0]]))
            m += m * 2. * margin
            axes.set_xlim(-m, m)
        else:
            m = (box.upper[ax[0]] - box.lower[ax[0]]) * margin
            axes.set_xlim(box.lower[ax[0]] - m, box.upper[ax[0]] + m)
        if vmirror:
            m = max(abs(box.lower[ax[1]]), abs(box.upper[ax[1]]))
            m += m * 2. * margin
            axes.set_ylim(-m, m)
        else:
            m = (box.upper[ax[1]] - box.lower[ax[1]]) * margin
            axes.set_ylim(box.lower[ax[1]] - m, box.upper[ax[1]] + m)

    if ax[0] > ax[1] and not axes.yaxis_inverted():
        axes.invert_yaxis()

    axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[0]]))
    axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[1]]))

    return axes
