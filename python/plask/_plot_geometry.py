# -*- coding: utf-8 -*-
# ### plot_geometry ###
import math
import colorsys

import plask
import matplotlib
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.artist

from numpy import array

import collections
from zlib import crc32

from .pylab import _get_2d_axes

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

    def __nonzero__(self):
        return not (self._bbox2.xmin > self._bbox1.xmax or
                    self._bbox2.xmax < self._bbox1.xmin or
                    self._bbox2.ymin > self._bbox1.ymax or
                    self._bbox2.ymax < self._bbox1.ymin)

    def get_points(self):
        if self._invalid:
            if self:
                x0 = max([self._bbox1.xmin, self._bbox2.xmin])
                x1 = min([self._bbox1.xmax, self._bbox2.xmax])
                y0 = max([self._bbox1.ymin, self._bbox2.ymin])
                y1 = min([self._bbox1.ymax, self._bbox2.ymax])
                self._points = [[x0, y0], [x1, y1]]
            else:
                self._points = matplotlib.transforms.Bbox.from_bounds(0, 0, -1, -1).get_points()
            self._invalid = 0
        return array(self._points)
    get_points.__doc__ = matplotlib.transforms.Bbox.get_points.__doc__


class ColorFromDict(object):
    """
    Get color from Python dictionary. The dictionary should map material name
    or regular expression object to RGB tuple with each component in range [0, 1].

    Args:
        material_dict (dict): Dictionary with mapping.
        axes (matplotlib.axes.Axes): Matplotlib axes, to which the geometry will
                                     plotted. It is used for retrieving background
                                     color.
    """

    def __init__(self, material_dict, axes=None):
        if material_dict is not DEFAULT_COLORS:
            self.default_color = ColorFromDict(DEFAULT_COLORS, axes)
            if any(isinstance(m, plask.material.Material) for m in material_dict):
                material_dict = dict((str(k), v) for k, v in material_dict.items())
        else:
            if axes is None:
                self._air_color = '#ffffff'
            else:
                try:
                    self._air_color = axes.get_facecolor()
                except AttributeError:
                    self._air_color = axes.get_axis_bgcolor()
            self.default_color = self.auto_color
        self.material_dict = material_dict

    def __call__(self, material):
        """
        Get color for specified material.

        Args:
            material (str): Material name

        Returns:
            tuple: Tuple with desired color in RGB format.
        """
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
            return self.default_color(material)

    def auto_color(self, material):
        """
            Generate color for given material.
            :param plask.Material material: material
            :return (float, float, float): RGB color, 3 floats, each in range [0, 1]
        """
        s = str(material)
        if s == 'air':
            return self._air_color
        i = crc32(s.encode('utf8'))
        h, s, v = (i & 0xff), (i >> 8) & 0xff, (i >> 16) & 0xff
        h, s, v = (h + 12.) / 279., (s + 153.) / 408., (v + 153.) / 408.
        return colorsys.hsv_to_rgb(h, s, v)


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

    'SiO2':                                 '#FFD699',
    'Si':                                   '#BF7300',
    'aSiO2':                                '#FFDF99',
    'aSi':                                  '#BF8300',

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

    def __init__(self, plane, dest, fill=False, color='k', get_color=None, lw=1.0, alpha=1.0, zorder=3.0, picker=None,
                 extra=None):
        """
        :param plane: plane to draw (important in 3D)
        :param dest: mpl axis where artist should be appended
        :param bool fill: True if artists should be filled
        :param color: edge color (mpl format)
        :param lw: line width
        :param alpha: opacity of the drawn environment
        :param zorder: artists z order
        :param extra: flag indicating if extra_patches should be taken
        """
        super(DrawEnviroment, self).__init__()
        self.dest = dest
        self.fill = fill
        self.color = color
        self.lw = lw
        self.alpha = alpha
        self.axes = plane
        self.zorder = zorder
        self.picker = picker
        self.extra_patches = {}
        self.extra = extra

        if get_color is None:
            self.get_color = ColorFromDict(DEFAULT_COLORS, dest)
        elif get_color is not None and not isinstance(get_color, collections.Callable):
            self.get_color = ColorFromDict(get_color, dest)
        else:
            self.get_color = get_color

    def append(self, artist, clipbox, geometry_object, plask_real_path=None):
        """
        Configure and append artist to destination axis object.
        :param artist: artist to append
        :param matplotlib.transforms.BboxBase clipbox: clipping box for artist, optional
        :param geometry_object: plask's geometry object which is represented by the artist
        """
        if self.fill and geometry_object is not None:
            artist.set_fill(True)
            artist.set_facecolor(self.get_color(geometry_object.representative_material))
        else:
            artist.set_fill(False)
        artist.set_linewidth(self.lw)
        artist.set_ec(self.color)
        artist.set_alpha(self.alpha)
        artist.set_picker(self.picker)
        artist.plask_real_path = plask_real_path
        self.dest.add_patch(artist)
        if clipbox is not None:
            artist.set_clip_box(BBoxIntersection(clipbox, artist.get_clip_box()))
            #artist.set_clip_box(clipbox)
            #artist.set_clip_on(True)
            #artist.set_clip_path(clipbox)
        artist.set_zorder(self.zorder)

    def append_extra(self, geometry_object, artist):
        """
        Configure and append artist to destination axis object.
        :param artist: artist to append
        :param matplotlib.transforms.BboxBase clipbox: clipping box for artist, optional
        :param geometry_object: plask's geometry object which is represented by the artist
        """
        artist.set_fill(False)
        for attr in self.extra:
            getattr(artist, 'set_'+attr)(self.extra[attr])
        self.extra_patches.setdefault(geometry_object, []).append(artist)

def _draw_bbox(env, geometry_object, bbox, transform, clipbox, plask_real_path):
    block = matplotlib.patches.Rectangle(
        (bbox.lower[env.axes[0]], bbox.lower[env.axes[1]]),
        bbox.upper[env.axes[0]]-bbox.lower[env.axes[0]], bbox.upper[env.axes[1]]-bbox.lower[env.axes[1]],
        transform=transform
    )
    env.append(block, clipbox, geometry_object, plask_real_path)


def _draw_Block(env, geometry_object, transform, clipbox, plask_real_path):
    _draw_bbox(env, geometry_object, geometry_object.bbox, transform, clipbox, plask_real_path)

_geometry_drawers[plask.geometry.Block2D] = _draw_Block
_geometry_drawers[plask.geometry.Block3D] = _draw_Block


def _draw_Triangle(env, geometry_object, transform, clipbox, plask_real_path):
    p1 = geometry_object.a
    p2 = geometry_object.b
    env.append(matplotlib.patches.Polygon(((0.0, 0.0), (p1[0], p1[1]), (p2[0], p2[1])), closed=True, transform=transform),
               clipbox, geometry_object, plask_real_path
    )

_geometry_drawers[plask.geometry.Triangle] = _draw_Triangle


def _draw_Circle(env, geometry_object, transform, clipbox, plask_real_path):
    env.append(matplotlib.patches.Circle((0.0, 0.0), geometry_object.radius, transform=transform),
               clipbox, geometry_object, plask_real_path
    )

_geometry_drawers[plask.geometry.Circle] = _draw_Circle
_geometry_drawers[plask.geometry.Sphere] = _draw_Circle


def _draw_Cylinder(env, geometry_object, transform, clipbox, plask_real_path):
    if env.axes == (0, 1) or env.axes == (1, 0):
        _draw_Circle(env, geometry_object, transform, clipbox, plask_real_path)
    else:
        _draw_Block(env, geometry_object, transform, clipbox, plask_real_path)

_geometry_drawers[plask.geometry.Cylinder] = _draw_Cylinder


def _draw_Prism(env, geometry_object, transform, clipbox, plask_real_path):
    p1 = geometry_object.a
    p2 = geometry_object.b
    if env.axes == (0, 1) or env.axes == (1, 0):
        env.append(matplotlib.patches.Polygon(
            ((0.0, 0.0), (p1[env.axes[0]], p1[env.axes[1]]), (p2[env.axes[0]], p2[env.axes[1]])),
            closed=True, transform=transform),
            clipbox, geometry_object, plask_real_path
        )
    else:
        axis = [a for a in env.axes if a != 2][0]
        pts = sorted((0., p1[axis], p2[axis]))
        height = geometry_object.height
        box = matplotlib.patches.Rectangle((pts[0], 0.), pts[-1] - pts[0], height, transform=transform)
        env.append(box, clipbox, geometry_object, plask_real_path)
        if pts[1] != pts[0] and pts[1] != pts[2]:
            line = matplotlib.patches.Polygon(((pts[1], 0.), (pts[1], height)), transform=transform)
            env.append(line, clipbox, geometry_object, plask_real_path)

        _draw_Block(env, geometry_object, transform, clipbox, plask_real_path)


_geometry_drawers[plask.geometry.Prism] = _draw_Prism


def _draw_Extrusion(env, geometry_object, transform, clipbox, plask_real_path):
    if env.axes == (1, 2) or env.axes == (2, 1):
        try:
            env.axes = tuple(x-1 for x in env.axes)  # change axes to 2D
            _draw_geometry_object(env, geometry_object.item, transform, clipbox, plask_real_path + [0])
        finally:    # revert axes settings, change back to 3D:
            env.axes = tuple(x+1 for x in env.axes)
    else:
        #_draw_Block(env, geometry_object, transform, clipbox)  # draw block uses bbox, so it will work fine
        for leaf_bbox in geometry_object.get_leafs_bboxes():
            _draw_bbox(env, None, leaf_bbox, transform, clipbox, plask_real_path)

_geometry_drawers[plask.geometry.Extrusion] = _draw_Extrusion


def _draw_Revolution(env, geometry_object, transform, clipbox, plask_real_path):
    if env.axes == (0, 1) or env.axes == (1, 0):    # view from the top
        obj2d = geometry_object.item
        rads = set()
        for bb in geometry_object.item.get_leafs_bboxes():
            rads.add(bb.left)
            rads.add(bb.right)
        for r in rads:
            if r > 0:
                env.append(matplotlib.patches.Circle((0.0, 0.0), r, transform=transform), clipbox, obj2d, plask_real_path)
    else:
        original_axes = env.axes
        env.axes = tuple(0 if x == 0 else x-1 for x in original_axes)
        try:    #TODO modify clip-box?
            new_plask_real_path = plask_real_path + [0]
            _draw_geometry_object(env, geometry_object.item, transform, clipbox, new_plask_real_path)
            _draw_Flipped(env, geometry_object.item, transform, clipbox, 0, new_plask_real_path)
            #_draw_Block(env, geometry_object, transform, clipbox)
        finally:
            env.axes = original_axes

_geometry_drawers[plask.geometry.Revolution] = _draw_Revolution


def _draw_Translation(env, geometry_object, transform, clipbox, plask_real_path):
    new_transform = matplotlib.transforms.Affine2D()
    t = geometry_object.vec
    new_transform.translate(t[env.axes[0]], t[env.axes[1]])
    _draw_geometry_object(env, geometry_object.item, new_transform + transform, clipbox, plask_real_path + [0])

_geometry_drawers[plask.geometry.Translation2D] = _draw_Translation
_geometry_drawers[plask.geometry.Translation3D] = _draw_Translation


def _draw_Lattice(env, geometry_object, transform, clipbox, plask_real_path):
    for index, child in enumerate(geometry_object):
        _draw_geometry_object(env, child, transform, clipbox, plask_real_path + [index])
    if env.extra is not None:
        v0, v1 = geometry_object.vec0, geometry_object.vec1
        for v in v0, v1:
            arrow = matplotlib.patches.FancyArrowPatch((0, 0), (v[env.axes[0]], v[env.axes[1]]),
                                                       arrowstyle='->', mutation_scale=40,
                                                       transform=transform)
            env.append_extra(geometry_object, arrow)
        for segment in geometry_object.segments:
            if not segment: continue
            polygon = [(p[env.axes[0]], p[env.axes[1]]) for p in (v0*a0+v1*a1 for (a0,a1) in segment)]
            env.append_extra(geometry_object, matplotlib.patches.Polygon(polygon, closed=True, transform=transform))

_geometry_drawers[plask.geometry.Lattice] = _draw_Lattice


def _draw_Flipped(env, geometry_object, transform, clipbox, axis_nr, plask_real_path):
    if axis_nr == env.axes[0]:
        _draw_geometry_object(env, geometry_object, matplotlib.transforms.Affine2D.from_values(-1.0, 0, 0, 1.0, 0, 0) + transform, clipbox, plask_real_path)
    elif axis_nr == env.axes[1]:
        _draw_geometry_object(env, geometry_object, matplotlib.transforms.Affine2D.from_values(1.0, 0, 0, -1.0, 0, 0) + transform, clipbox, plask_real_path)
    else:
        _draw_geometry_object(env, geometry_object, transform, clipbox, plask_real_path)


def _draw_Flip(env, geometry_object, transform, clipbox, plask_real_path):
    _draw_Flipped(env, geometry_object.item, transform, clipbox, geometry_object.axis_nr, plask_real_path + [0])

_geometry_drawers[plask.geometry.Flip2D] = _draw_Flip
_geometry_drawers[plask.geometry.Flip3D] = _draw_Flip


def _draw_Mirror(env, geometry_object, transform, clipbox, plask_real_path):
    #TODO modify clip-box?
    _draw_geometry_object(env, geometry_object.item, transform, clipbox, plask_real_path + [0])
    if geometry_object.axis_nr in env.axes:  # in 3D this must not be true
        _draw_Flip(env, geometry_object, transform, clipbox, plask_real_path)

_geometry_drawers[plask.geometry.Mirror2D] = _draw_Mirror
_geometry_drawers[plask.geometry.Mirror3D] = _draw_Mirror


def _draw_clipped(env, geometry_object, transform, clipbox, new_clipbox, plask_real_path):
    """Used by _draw_Clip and _draw_Intersection."""
    def _b(bound):
        return math.copysign(1e100, bound) if math.isinf(bound) else bound

    new_clipbox = matplotlib.transforms.TransformedBbox(
       #matplotlib.transforms.Bbox([
       #    [obj_box.lower[env.axes[0]], obj_box.lower[env.axes[1]]],
       #    [obj_box.upper[env.axes[0]], obj_box.upper[env.axes[1]]]
       #]),
       matplotlib.transforms.Bbox.from_extents(_b(new_clipbox.lower[env.axes[0]]), _b(new_clipbox.lower[env.axes[1]]),
                                               _b(new_clipbox.upper[env.axes[0]]), _b(new_clipbox.upper[env.axes[1]])),
       transform
    )

    if clipbox is None:
        clipbox = new_clipbox
    else:
        clipbox = BBoxIntersection(clipbox, new_clipbox)

    if clipbox:
        _draw_geometry_object(env, geometry_object, transform, clipbox, plask_real_path)
    # else, if clipbox is empty now, it will be never non-empty, so all will be clipped-out


def _draw_Clip(env, geometry_object, transform, clipbox, plask_real_path):
    _draw_clipped(env, geometry_object.item, transform, clipbox, geometry_object.clipbox, plask_real_path + [0])

_geometry_drawers[plask.geometry.Clip2D] = _draw_Clip
_geometry_drawers[plask.geometry.Clip3D] = _draw_Clip


def _draw_Intersection(env, geometry_object, transform, clipbox, plask_real_path):
    if geometry_object.envelope is not None:
        _draw_clipped(env, geometry_object.item, transform, clipbox, geometry_object.envelope.bbox, plask_real_path + [0])
    else:
        _draw_geometry_object(env, geometry_object.item, transform, clipbox, plask_real_path + [0])

_geometry_drawers[plask.geometry.Intersection2D] = _draw_Intersection
_geometry_drawers[plask.geometry.Intersection3D] = _draw_Intersection


def _draw_geometry2d(env, geometry_object, transform, clipbox, plask_real_path):
    _draw_geometry_object(env, geometry_object.item, transform, clipbox, plask_real_path + [0, 0])

_geometry_drawers[plask.geometry.Cartesian2D] = _draw_geometry2d
_geometry_drawers[plask.geometry.Cylindrical2D] = _draw_geometry2d



def _draw_geometry_object(env, geometry_object, transform, clipbox, plask_real_path=None):
    """
    Draw geometry object.
    :param DrawEnviroment env: drawing configuration
    :param geometry_object: object to draw
    :param transform: transform from a plot coordinates to the geometry_object
    :param matplotlib.transforms.BboxBase clipbox: clipping box in plot coordinates
    """
    if geometry_object is None: return
    if plask_real_path is None: plask_real_path = []
    drawer = _geometry_drawers.get(type(geometry_object))
    if drawer is None:
        try:
            for index, child in enumerate(geometry_object):
                _draw_geometry_object(env, child, transform, clipbox, plask_real_path + [index])
            #for child in geometry_object:
            #    _draw_geometry_object(env, child, transform, clipbox)
        except TypeError:
            pass    # ignore non-iterable object
    else:
        drawer(env, geometry_object, transform, clipbox, plask_real_path)


def plane_to_axes(plane, dim):
    """
    Get number of axes used by plot_geometry.
    :param str plane: plane parameter given to plot_geometry, a string with two letters specifying axis names of the drawn plane.
    :param int dim: number of dimension of geometry object passed to plot_geometry, 2 or 3
    :return (int, int): numbers of the first and the second axis
    """
    return _get_2d_axes(plane) if dim == 3 else (0, 1)


def plot_geometry(geometry, color='k', lw=1.0, plane=None, zorder=None, mirror=False, periods=(1,1), fill=False,
                  axes=None, figure=None, margin=None, get_color=None, alpha=1.0, extra=None, picker=None):
    """
    Plot specified geometry.

    Args:
        geometry (plask.geometry.Geometry): Geometry to draw.

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
                specification says so (i.e. some edges are set to
                *mirror* or the geometry is a cylindrical one).

        periods (int): Number of periods to plot periodic geometries.

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

        get_color (callable or dict): Callable that gets color for material given
                as its parameter or dictionary from material names (strings)
                to colors. Material color should be given as a triple
                (float, float, float) of red, green, blue components, each in
                range [0, 1]. Any other format accepted by set_facecolor()
                method of matplotlib Artist should work as well.

        alpha (float): Opacity of the drawn geometry (1: fully opaque,
                0: fully transparent)

        extra (None|dict): If this parameter is not None, a dictionary with optional
                extra patches for some geometry objects is returned in addition
                to axes. In such case this parameter value must be a dict with extra
                patches style (with keys like 'edgeceolor', 'linewidth', etc.,
                see Matplotlib documentation for details).

        picker (None|float|boolean|callable) matplotlib picker attribute
                for all artists appended to plot (see matplotlib doc.).

    Returns:
        matplotlib.axes.Axes: appended or given axes object

        dict (optional): dictionary mapping geometry objects to extra_patches.

    Limitations:
        Intersection is not drawn precisely (item is clipped to bounding box of
        the envelope).

        Filling is not supported when 3D geometry object or Cartesian3D geometry is drawn.
    """

    if axes is None:
        if figure is None:
            axes = matplotlib.pylab.gca()
        else:
            axes = figure.add_subplot(111)

    # if isinstance(geometry, plask.geometry.Cartesian3D):
    if geometry.dims == 3:
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

    env = DrawEnviroment(ax, axes, fill, color, get_color, lw, alpha, zorder=zorder, picker=picker,
                         extra=extra)

    hshift, vshift = (geometry.bbox.size[a] for a in ax)
    try:
        periods = array((periods[0], periods[1]), int)
    except TypeError:
        periods = array((periods, periods), int)
    try:
        if geometry.edges[dirs[0][0]] == 'mirror' or geometry.edges[dirs[0][1]] == 'mirror' or \
           isinstance(geometry, plask.geometry.Cylindrical2D):
            hshift *= 2
            hmirrortransform = matplotlib.transforms.Affine2D.from_values(-1., 0, 0, 1., 0, 0)
            hmirror = mirror
            periods[0] = 2*periods[0] - 1
        else:
            hmirror = False
        if geometry.edges[dirs[1][0]] == 'mirror' or geometry.edges[dirs[1][1]] == 'mirror':
            vshift *= 2
            vmirrortransform = matplotlib.transforms.Affine2D.from_values(1., 0, 0, -1., 0, 0)
            vmirror = mirror
            periods[1] = 2*periods[1] - 1
            if hmirror:
                vhmirrortransform = matplotlib.transforms.Affine2D.from_values(-1., 0, 0, -1., 0, 0)
        else:
            vmirror = False
        if geometry.edges[dirs[0][0]] == 'periodic' or geometry.edges[dirs[0][1]] == 'periodic':
            hstart = -int((periods[0]-1) / 2)
            hrange = range(hstart, hstart + max(periods[0], 1))
        else:
            hrange = (0,)
        if geometry.edges[dirs[1][0]] == 'periodic' or geometry.edges[dirs[1][1]] == 'periodic':
            vstart = -int((periods[1]-1) / 2)
            vrange = range(vstart, vstart + max(periods[1], 1))
        else:
            vrange = (0,)
    except AttributeError:  # we draw non-Geometry object
        hmirror = False
        vmirror = False
        hrange = (0,)
        vrange = (0,)

    for iv in vrange:
        for ih in hrange:
            shift = matplotlib.transforms.Affine2D()
            shift.translate(ih*hshift, iv*vshift)
            _draw_geometry_object(env, geometry, shift + axes.transData, None)
            if hmirror:
                _draw_geometry_object(env, geometry,
                                      shift + hmirrortransform + axes.transData, None)
            if vmirror:
                _draw_geometry_object(env, geometry,
                                      shift + vmirrortransform + axes.transData, None)
                if hmirror:
                    _draw_geometry_object(env, geometry,
                                          shift + vhmirrortransform + axes.transData, None)

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

    if extra is not None:
        return axes, env.extra_patches
    else:
        return axes
