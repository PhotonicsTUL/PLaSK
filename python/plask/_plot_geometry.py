# -*- coding: utf-8 -*-
# ### plot_geometry ###
import math

import plask
import matplotlib
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.artist

from numpy import array

import collections
from zlib import adler32

from pylab import _get_2d_axes

from re import compile as _r

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
            intersects = not (self._bbox2.xmin > self._bbox1.xmax or
                              self._bbox2.xmax < self._bbox1.xmin or
                              self._bbox2.ymin > self._bbox1.ymax or
                              self._bbox2.ymax < self._bbox1.ymin)
            if intersects:
                x0 = max([self._bbox1.xmin, self._bbox2.xmin])
                x1 = min([self._bbox1.xmax, self._bbox2.xmax])
                y0 = max([self._bbox1.ymin, self._bbox2.ymin])
                y1 = min([self._bbox1.ymax, self._bbox2.ymax])
                self._points = [[x0, y0], [x1, y1]]
            else:
                self._points = matplotlib.transforms.Bbox.from_bounds(0, 0, -1, -1).get_points()
            self._invalid = 0
        return self._points
    get_points.__doc__ = matplotlib.transforms.Bbox.get_points.__doc__


class _MaterialToColor(object):

    def __init__(self, axes=None):
        if axes is None:
            self._air_color = '#ffffff'
        else:
            self._air_color = axes.get_axis_bgcolor()

    def __call__(self, material):
        """
            Generate color for given material.
            :param plask.Material material: material
            :return (float, float, float): RGB color, 3 floats, each in range [0, 1]
        """
        s = str(material)
        if s == 'air':
            return self._air_color
        i = adler32(s)      # maybe crc32?
        return (i & 0xff) / 255.0, ((i >> 8) & 0xff) / 255.0, ((i >> 16) & 0xff) / 255.0


class _ColorFromDict(object):
    """
    Get color from dict:
    material name string/re -> color or using material_to_color (for names which are not present in dict).
    """

    def __init__(self, material_dict, axes=None):
        super(_ColorFromDict, self).__init__()
        self.material_dict = material_dict
        self.material_to_color = _MaterialToColor(axes)

    def __call__(self, material):
        s = str(material)
        try:
            return self.material_dict[s]
        except KeyError:
            for r in self.material_dict:
                try:
                    m = r.match(s)
                    if m is not None:
                        c = self.material_dict[r]
                        if isinstance(c, collections.Callable):
                            return c(*m.groups())
                        else:
                            return m.expand(c)
                except AttributeError:
                    pass
            return self.material_to_color(material)


class TertiaryColors(object):
    """
    Interpolate colors of tertiary compounds.
    """

    def __init__(self, color1, color2):
        self.color2 = array(color2)
        self.color12 = array(color1) - self.color2

    def __call__(self, x):
        return self.color2 + float(x) * self.color12


class DopedColors(object):
    """
    Modify  color by doping
    """

    def __init__(self, color, doping_color):
        self.color = color
        self.doping_color = array(doping_color)

    def __call__(self, *args):
        if isinstance(self.color, collections.Callable):
            result = self.color(*args[:-1])
        else:
            result = array(self.color)
        result += self.doping_color * math.log10(float(args[-1]))/20.
        result[result > 1.] = 1.
        result[result < 0.] = 0.
        return result


# Default colors

_GaAs = (0.00, 0.62, 0.00)
_AlAs = (0.82, 0.94, 0.00)
_InAs = (0.82, 0.50, 0.00)
_AlGaAs = TertiaryColors(_AlAs, _GaAs)
_InGaAs = TertiaryColors(_InAs, _GaAs)
_As_n = (0.0, 0.0, 0.3)
_As_p = (0.1, 0.0, 0.0)

_GaN = (0.00, 0.00, 0.62)
_AlN = (0.00, 0.82, 0.94)
_InN = (0.82, 0.00, 0.50)
_AlGaN = TertiaryColors(_AlN, _GaN)
_InGaN = TertiaryColors(_InN, _GaN)
_N_n = (0.0, 0.2, 0.0)
_N_p = (0.2, 0.0, 0.0)

DEFAULT_COLORS = {
    'Cu':                                   '#9E807E',
    'Au':                                   '#A6A674',
    'Pt':                                   '#A6A674',
    'In':                                   '#585266',

    'AlOx':                                 '#98F2FF',

    'GaAs':                                 _GaAs,
    _r(r'GaAs:Si.*=(.*)'):                  DopedColors(_GaAs, _As_n),
    _r(r'GaAs:(?:Be|Zn|C).*=(.*)'):         DopedColors(_GaAs, _As_p),
    'AlAs':                                 _AlAs,
    _r(r'AlAs:Si.*=(.*)'):                  DopedColors(_AlAs, _As_n),
    _r(r'AlAs:C.*=(.*)'):                   DopedColors(_AlAs, _As_p),
    'InAs':                                 _InAs,
    _r(r'InAs:Si.*=(.*)'):                  DopedColors(_InAs, _As_n),
    _r(r'InAs:C.*=(.*)'):                   DopedColors(_InAs, _As_p),
    _r(r'Al\(([\d.]+)\)GaAs$'): _AlGaAs,
    _r(r'Al\(([\d.]+)\)GaAs:Si.*=(.*)'):    DopedColors(_AlGaAs, _As_n),
    _r(r'Al\(([\d.]+)\)GaAs:C.*=(.*)'):     DopedColors(_AlGaAs, _As_p),
    _r(r'In\(([\d.]+)\)GaAs$'): _InGaAs,
    _r(r'In\(([\d.]+)\)GaAs:Si.*=(.*)'):    DopedColors(_InGaAs, _As_n),
    _r(r'In\(([\d.]+)\)GaAs:C.*=(.*)'):     DopedColors(_InGaAs, _As_p),

    'GaN':                                  _GaN,
    'GaN_bulk':                             (0.00, 0.00, 0.50),
    _r(r'GaN:Si.*=(.*)'):                   DopedColors(_GaN, _N_n),
    _r(r'GaN:Mg.*=(.*)'):                   DopedColors(_GaN, _N_p),
    'AlN':                                  _AlN,
    _r(r'AlN:Si.*=(.*)'):                   DopedColors(_AlN, _N_n),
    _r(r'AlN:Mg.*=(.*)'):                   DopedColors(_AlN, _N_p),
    'InN':                                  _InN,
    _r(r'InN:Si.*=(.*)'):                   DopedColors(_InN, _N_n),
    _r(r'InN:C.*=(.*)'):                    DopedColors(_InN, _N_p),
    _r(r'Al\(([\d.]+)\)GaN$'):              _AlGaN,
    _r(r'Al\(([\d.]+)\)GaN:Si.*=(.*)'):     DopedColors(_AlGaN, _N_n),
    _r(r'Al\(([\d.]+)\)GaN:Mg.*=(.*)'):     DopedColors(_AlGaN, _N_p),
    _r(r'In\(([\d.]+)\)GaN$'):              _InGaN,
    _r(r'In\(([\d.]+)\)GaN:Si.*=(.*)'):     DopedColors(_InGaN, _N_n),
    _r(r'In\(([\d.]+)\)GaN:Mg.*=(.*)'):     DopedColors(_InGaN, _N_p),
}


class DrawEnviroment(object):
    """
        Drawing configuration.
    """

    def __init__(self, plane, dest, fill=False, color='k', get_color=None, lw=1.0, zorder=3.0):
        """
        :param plane: plane to draw (important in 3D)
        :param dest: mpl axis where artist should be appended
        :param bool fill: True if artists should be filled
        :param color: edge color (mpl format)
        :param lw: line width
        :param zorder: artists z order
        """
        super(DrawEnviroment, self).__init__()
        self.dest = dest
        self.fill = fill
        self.color = color
        self.lw = lw
        self.axes = plane
        self.zorder = zorder

        if get_color is None:
            self.get_color = _ColorFromDict(DEFAULT_COLORS, dest)
        elif get_color is not None and not isinstance(get_color, collections.Callable):
            self.get_color = _ColorFromDict(get_color, dest)
        else:
            self.get_color = get_color

    def append(self, artist, clip_box, geometry_object):
        """
        Configure and append artist to destination axis object.
        :param artist: artist to append
        :param matplotlib.transforms.BboxBase clip_box: clipping box for artist, optional
        :param geometry_object: plask's geometry object which is represented by the artist
        """
        if self.fill and geometry_object is not None:
            artist.set_fill(True)
            artist.set_facecolor(self.get_color(geometry_object.representative_material))
        else:
            artist.set_fill(False)
        artist.set_linewidth(self.lw)
        artist.set_ec(self.color)
        self.dest.add_patch(artist)
        if clip_box is not None:
            artist.set_clip_box(BBoxIntersection(clip_box, artist.get_clip_box()))
            #artist.set_clip_box(clip_box)
            #artist.set_clip_on(True)
            #artist.set_clip_path(clip_box)
        artist.set_zorder(self.zorder)


def _draw_bbox(env, geometry_object, bbox, transform, clip_box):
    block = matplotlib.patches.Rectangle(
        (bbox.lower[env.axes[0]], bbox.lower[env.axes[1]]),
        bbox.upper[env.axes[0]]-bbox.lower[env.axes[0]], bbox.upper[env.axes[1]]-bbox.lower[env.axes[1]],
        transform=transform
    )
    env.append(block, clip_box, geometry_object)


def _draw_Block(env, geometry_object, transform, clip_box):
    _draw_bbox(env, geometry_object, geometry_object.bbox, transform, clip_box)

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
    if env.axes == (0, 1) or env.axes == (1, 0):
        _draw_Circle(env, geometry_object, transform, clip_box)
    else:
        _draw_Block(env, geometry_object, transform, clip_box)

_geometry_drawers[plask.geometry.Cylinder] = _draw_Cylinder


def _draw_Extrusion(env, geometry_object, transform, clip_box):
    if env.axes == (1, 2) or env.axes == (2, 1):
        try:
            env.axes = tuple(x-1 for x in env.axes)  # change axes to 2D
            _draw_geometry_object(env, geometry_object.item, transform, clip_box)
        finally:    # revert axes settings, change back to 3D:
            env.axes = tuple(x+1 for x in env.axes)
    else:
        #_draw_Block(env, geometry_object, transform, clip_box)  # draw block uses bbox, so it will work fine
        for leaf_bbox in geometry_object.get_leafs_bboxes():
            _draw_bbox(env, None, leaf_bbox, transform, clip_box)



_geometry_drawers[plask.geometry.Extrusion] = _draw_Extrusion


def _draw_Revolution(env, geometry_object, transform, clip_box):
    if env.axes == (0, 1) or env.axes == (1, 0):    # view from the top
        obj2d = geometry_object.item
        bbox = obj2d.bbox
        env.append(matplotlib.patches.Circle((0.0, 0.0), bbox.upper[0], transform=transform), clip_box, obj2d)
        if bbox.lower[0] > 0:
            env.append(matplotlib.patches.Circle((0.0, 0.0), bbox.lower[0], transform=transform), clip_box, obj2d)
    else:
        original_axes = env.axes
        env.axes = tuple(0 if x == 0 else x-1 for x in original_axes)
        try:    #TODO modify clip-box?
            _draw_geometry_object(env, geometry_object.item, transform, clip_box)
            _draw_Flipped(env, geometry_object.item, transform, clip_box, 0)
            #_draw_Block(env, geometry_object, transform, clip_box)
        finally:
            env.axes = original_axes


_geometry_drawers[plask.geometry.Revolution] = _draw_Revolution


def _draw_Translation(env, geometry_object, transform, clip_box):
    new_transform = matplotlib.transforms.Affine2D()
    t = geometry_object.translation
    new_transform.translate(t[env.axes[0]], t[env.axes[1]])
    _draw_geometry_object(env, geometry_object.item, new_transform + transform, clip_box)

_geometry_drawers[plask.geometry.Translation2D] = _draw_Translation
_geometry_drawers[plask.geometry.Translation3D] = _draw_Translation


def _draw_Flipped(env, geometry_object, transform, clip_box, axis_nr):
    if axis_nr == env.axes[0]:
        _draw_geometry_object(env, geometry_object, matplotlib.transforms.Affine2D.from_values(-1.0, 0, 0, 1.0, 0, 0) + transform, clip_box)
    elif axis_nr == env.axes[1]:
        _draw_geometry_object(env, geometry_object, matplotlib.transforms.Affine2D.from_values(1.0, 0, 0, -1.0, 0, 0) + transform, clip_box)
    else:
        _draw_geometry_object(env, geometry_object, transform, clip_box)

def _draw_Flip(env, geometry_object, transform, clip_box):
    _draw_Flipped(env, geometry_object.item, transform, clip_box, geometry_object.axis_nr)

_geometry_drawers[plask.geometry.Flip2D] = _draw_Flip
_geometry_drawers[plask.geometry.Flip3D] = _draw_Flip


def _draw_Mirror(env, geometry_object, transform, clip_box):
    #TODO modify clip-box?
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
            pass    # ignore non-iterable object
    else:
        drawer(env, geometry_object, transform, clip_box)


def plot_geometry(geometry, color='k', lw=1.0, plane=None, zorder=None, mirror=False, fill=False,
                  axes=None, figure=None, margin=None, get_color=None, set_limits=None):
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

        Filling is not supported when 3D geometry object or Cartesian3D geometry is drawn.
    """

    if set_limits is not None:
        plask.print_log('warning', "plot_geometry: 'set_limits' is obsolette, set 'margin' instead")
        if margin is None:
            margin = 0.

    if axes is None:
        if figure is None:
            axes = matplotlib.pylab.gca()
        else:
            axes = figure.add_subplot(111)

    # if isinstance(geometry, plask.geometry.Cartesian3D):
    if geometry.DIMS == 3:
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

    if zorder is None:
        zorder = 0.5 if fill else 2.0

    env = DrawEnviroment(ax, axes, fill, color, get_color, lw, zorder=zorder)

    try:
        hmirror = mirror and (geometry.borders[dirs[0][0]] == 'mirror' or geometry.borders[dirs[0][1]] == 'mirror' or
                              type(geometry) == plask.geometry.Cylindrical2D)
        vmirror = mirror and (geometry.borders[dirs[1][0]] == 'mirror' or geometry.borders[dirs[1][1]] == 'mirror')
    except AttributeError:  # we draw non-Geometry object
        hmirror = False
        vmirror = False

    _draw_geometry_object(env, geometry, axes.transData, None)
    if hmirror:
        _draw_geometry_object(env, geometry,
                              matplotlib.transforms.Affine2D.from_values(-1.0, 0, 0, 1.0, 0, 0) + axes.transData, None)
    if vmirror:
        _draw_geometry_object(env, geometry,
                              matplotlib.transforms.Affine2D.from_values(1.0, 0, 0, -1.0, 0, 0) + axes.transData, None)

    if margin is not None:
        box = geometry.bbox
        if hmirror:
            m = max(abs(box.lower[ax[0]]), abs(box.upper[ax[0]]))
            m += 2. * m * margin
            axes.set_xlim(-m, m)
        else:
            m = (box.upper[ax[0]] - box.lower[ax[0]]) * margin
            axes.set_xlim(box.lower[ax[0]] - m, box.upper[ax[0]] + m)
        if vmirror:
            m = max(abs(box.lower[ax[1]]), abs(box.upper[ax[1]]))
            m += 2. * m * margin
            axes.set_ylim(-m, m)
        else:
            m = (box.upper[ax[1]] - box.lower[ax[1]]) * margin
            axes.set_ylim(box.lower[ax[1]] - m, box.upper[ax[1]] + m)

    if ax[0] > ax[1] and not axes.yaxis_inverted():
        axes.invert_yaxis()

    axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[0]]))
    axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[dd + ax[1]]))

    return axes
