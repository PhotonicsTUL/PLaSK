# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# ### plot_geometry ###
import math
import colorsys
from copy import copy
import re

import plask
import matplotlib
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.artist

from numpy import array

from collections.abc import Callable
from zlib import crc32

from .pylab import _get_2d_axes

__all__ = ('plot_geometry')

to_rgb = matplotlib.colors.colorConverter.to_rgb


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
        return "BBoxIntersection(%r, %r)" % (self._bbox1, self._bbox2)

    def __bool__(self):
        return not (self._bbox2.xmin > self._bbox1.xmax or
                    self._bbox2.xmax < self._bbox1.xmin or
                    self._bbox2.ymin > self._bbox1.ymax or
                    self._bbox2.ymax < self._bbox1.ymin)
    __nonzero__ = __bool__  # for Python 2

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


class TertiaryColors:
    """
    Interpolate colors of tertiary compounds.
    """

    def __init__(self, color1, color2, x=None):
        self.color2 = array(to_rgb(color2))
        self.color12 = array(to_rgb(color1)) - self.color2
        self.x = x

    def __call__(self, x=None, **kwargs):
        if x is None and self.x is not None:
            x = kwargs[self.x]
        return self.color2 + float(x) * self.color12


class ColorFromDict:
    """
    Get color from Python dictionary. The dictionary should map material name
    or regular expression object to RGB tuple with each component in range [0, 1].

    Args:
        material_dict (dict): Dictionary with mapping.
        axes (matplotlib.axes.Axes): Matplotlib axes, to which the geometry will
                                     plotted. It is used for retrieving background
                                     color.
        tint_doping: Automatically add tint to doped materials.
        tertiary: Autmatically create combinations for these tertiary materials.
    """

    DOPING_TINTS = dict(N=array([0.0, 0.2, 0.2]), P=array([0.2, 0.0, 0.0]))

    def __init__(self, material_dict, axes=None, tint_doping=True,
                 tertiary=(('In', 'Al', 'Ga'), ('As', 'N', 'P', 'Sb'))):
        self.tint_doping = tint_doping
        if material_dict is not plask.MATERIAL_COLORS:
            self.default_color = ColorFromDict(plask.MATERIAL_COLORS, axes)
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
        self.tint_doping = tint_doping
        if tertiary is not None:
            self.material_dict = copy(self.material_dict)
            # TODO: add quarternary materials as well
            for third in tertiary[1]:
                for n, first in enumerate(tertiary[0][:-1]):
                    for second in tertiary[0][n+1:]:
                        try:
                            first_third = self.material_dict[first + third]
                        except KeyError:
                            first_third = self.default_color(first + third)
                        try:
                            second_third = self.material_dict[second + third]
                        except KeyError:
                            second_third = self.default_color(second + third)
                        colors = TertiaryColors(first_third, second_third, first)
                        r = re.compile(r'{}\(([\d.]+)\){}{}(?:_.*)?$'.format(first, second, third))
                        self.material_dict[r] = colors
                        # self.material_dict[frozenset((first, second, third))] = colors

    def __call__(self, material):
        """
        Get color for specified material.

        Args:
            material (plask.material.Material or str): Material.

        Returns:
            tuple: Tuple with desired color in RGB format.
        """
        s = m = str(material).split('[')[0].strip()
        if self.tint_doping:
            s = s.split(':')[0]
        try:
            result = self.material_dict[s]
        except KeyError:
            for r in self.material_dict:
                if isinstance(r, frozenset):
                    try:
                        if not isinstance(material, plask.material.Material):
                            material = plask.material.get(m)
                        if r == frozenset(material.composition):
                            c = self.material_dict[r]
                            result = c(**material.composition)
                            break
                    except:
                        pass
                else:  # regexp
                    try:
                        m = r.match(s)
                    except AttributeError:
                        pass
                    else:
                        if m is not None:
                            c = self.material_dict[r]
                            if isinstance(c, Callable):
                                result = c(*m.groups())
                            else:
                                result = m.expand(c)
                            break
            else:
                result = self.default_color(material)
        if self.tint_doping:
            try:
                if not isinstance(material, plask.material.Material):
                    material = plask.material.get(m)
                tint = self.DOPING_TINTS[material.condtype]
                doping = material.doping
            except:
                pass
            else:
                if doping > 0:
                    result = array(to_rgb(result))
                    result += tint * max(math.log10(doping)-15., 0.) / 5.
                    result[result > 1.] = 1.
                    result[result < 0.] = 0.
        return result

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


class Plane(matplotlib.patches.Patch):

    def __init__(self, x0=None, y0=None, x1=None, y1=None, **kwargs):
        self._x0 = x0
        self._y0 = y0
        self._x1 = x1
        self._y1 = y1
        super().__init__(**kwargs)

    def get_path(self):
        if self._x0 is None or self._x1 is None:
            x0, x1 = self.axes.get_xlim()
            if x0 > x1: x0, x1 = x1, x0
            dx = 0.01 * (x1 - x0)
            x0 = self._x0 if self._x0 is not None else x0 - dx
            x1 = self._x1 if self._x1 is not None else x1 + dx
        else:
            x0 = self._x0
            x1 = self._x1
        if self._y0 is None or self._y1 is None:
            y0, y1 = self.axes.get_ylim()
            if y0 > y1: y0, y1 = y1, y0
            dy = 0.01 * (y1 - y0)
            y0 = self._y0 if self._y0 is not None else y0 - dy
            y1 = self._y1 if self._y1 is not None else y1 + dy
        else:
            y0 = self._y0
            y1 = self._y1
        return matplotlib.patches.Path([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
                                       closed=True, readonly=True)

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, val):
        self._x0 = val
        self.stale = True

    @property
    def y0(self):
        return self._y0

    @y0.setter
    def y0(self, val):
        self._y0 = val
        self.stale = True

    @property
    def x1(self):
        return self._x1

    @x1.setter
    def x1(self, val):
        self._x1 = val
        self.stale = True

    @property
    def y1(self):
        return self._y1

    @y1.setter
    def y1(self, val):
        self._y1 = val
        self.stale = True


class PeriodicArtist(matplotlib.artist.Artist):

    def __init__(self, child, axes, clipbox, dx=0, dy=0):
        super().__init__()
        self._child = child
        self.set_zorder(child.get_zorder())
        self._dx = dx
        self._dy = dy
        (self._x0, self._y0), (self._x1, self._y1) = child.get_path().get_extents(child.get_patch_transform()).get_points()
        to_data = axes.transData.inverted()
        self._child_transform = (child.get_data_transform() + to_data).frozen()
        if clipbox is not None:
            self._clipbox = matplotlib.transforms.Bbox(to_data.transform(clipbox.get_points()))
        else:
            self._clipbox = None

    @staticmethod
    def _get_range(lims, x0, x1, dx):
        if dx == 0:
            return (0,)
        lo, hi = lims
        if lo > hi: lo, hi = hi, lo
        return range(math.floor((lo - x1) / dx), math.ceil((hi - x0) / dx) + 1)

    def draw(self, renderer):
        xrange = self._get_range(self.axes.get_xlim(), self._x0, self._x1, self._dx)
        yrange = self._get_range(self.axes.get_ylim(), self._y0, self._y1, self._dy)
        for iy in yrange:
            dy = iy * self._dy
            for ix in xrange:
                if ix == 0 and iy == 0: continue    # skip original element
                dx = ix * self._dx
                transl = matplotlib.transforms.Affine2D().translate(dx, dy)
                transform = transl + self.axes.transData
                self._child.set_transform(self._child_transform + transform)
                if self._clipbox is None:
                    self._child.set_clip_box(self.get_clip_box())
                else:
                    clipbox = matplotlib.transforms.TransformedBbox(self._clipbox, transform)
                    self._child.set_clip_box(BBoxIntersection(self.get_clip_box(), clipbox))
                self._child.draw(renderer)


class DrawEnviroment:
    """
        Drawing configuration.
    """

    def __init__(self, plane, dest, fill=False, color=None, get_color=None, lw=1.0, alpha=1.0, zorder=3.0, picker=None,
                 extra=None, periodic=None):
        """
        :param plane: plane to draw (important in 3D)
        :param dest: mpl axis where artist should be appended
        :param bool fill: True if artists should be filled
        :param color: edge color (mpl format)
        :param lw: line width
        :param alpha: opacity of the drawn environment
        :param zorder: artists z order
        :param extra: flag indicating if extra_patches should be taken
        :param periodic: None or arguments for periodic aritist
        """
        super().__init__()
        self.dest = dest
        self.fill = fill
        self.color = color if color is not None else matplotlib.rcParams['axes.edgecolor']
        self.lw = lw
        self.alpha = alpha
        self.axes = plane
        self.zorder = zorder
        self.picker = picker
        self.extra_patches = {}
        self.extra = extra
        self.periodic = periodic

        if get_color is None:
            self.get_color = ColorFromDict(plask.MATERIAL_COLORS, dest)
        elif get_color is not None and not isinstance(get_color, Callable):
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
        if self.periodic is not None:
            wrapper = PeriodicArtist(artist, self.dest, clipbox, **self.periodic)
            self.dest.add_artist(wrapper)
        else:
            self.dest.add_patch(artist)
            if clipbox is not None:
                artist.set_clip_box(BBoxIntersection(clipbox, artist.get_clip_box()))
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

def draw_bbox(env, geometry_object, bbox, transform, clipbox, plask_real_path):
    block = matplotlib.patches.Rectangle(
        (bbox.lower[env.axes[0]], bbox.lower[env.axes[1]]),
        bbox.upper[env.axes[0]]-bbox.lower[env.axes[0]], bbox.upper[env.axes[1]]-bbox.lower[env.axes[1]],
        transform=transform
    )
    env.append(block, clipbox, geometry_object, plask_real_path)


def draw_Block(env, geometry_object, transform, clipbox, plask_real_path):
    draw_bbox(env, geometry_object, geometry_object.bbox, transform, clipbox, plask_real_path)

def draw_Cuboid(env, geometry_object, transform, clipbox, plask_real_path):
    if (env.axes == (0, 1) or env.axes == (1, 0)) and geometry_object.angle is not None:
        rotation = matplotlib.transforms.Affine2D()
        rotation.rotate_deg(-geometry_object.angle if env.axes == (1, 0) else geometry_object.angle)
        new_transform = rotation + transform
        block = matplotlib.patches.Rectangle(
            (0, 0),
            geometry_object.dims[env.axes[0]], geometry_object.dims[env.axes[1]],
            transform=new_transform)
        env.append(block, clipbox, geometry_object, plask_real_path)
    else:
        draw_Block(env, geometry_object, transform, clipbox, plask_real_path)

_geometry_drawers[plask.geometry.Block2D] = draw_Block
_geometry_drawers[plask.geometry.Block3D] = draw_Cuboid
_geometry_drawers[plask.geometry._CuboidRotated] = draw_Cuboid


def draw_Triangle(env, geometry_object, transform, clipbox, plask_real_path):
    p1 = geometry_object.a
    p2 = geometry_object.b
    env.append(matplotlib.patches.Polygon(((0.0, 0.0), (p1[0], p1[1]), (p2[0], p2[1])), closed=True, transform=transform),
               clipbox, geometry_object, plask_real_path
    )

_geometry_drawers[plask.geometry.Triangle] = draw_Triangle


def draw_Circle(env, geometry_object, transform, clipbox, plask_real_path):
    env.append(matplotlib.patches.Circle((0.0, 0.0), geometry_object.radius, transform=transform),
               clipbox, geometry_object, plask_real_path
    )

_geometry_drawers[plask.geometry.Circle] = draw_Circle
_geometry_drawers[plask.geometry.Sphere] = draw_Circle


def draw_Cylinder(env, geometry_object, transform, clipbox, plask_real_path):
    if env.axes == (0, 1) or env.axes == (1, 0):
        draw_Circle(env, geometry_object, transform, clipbox, plask_real_path)
    else:
        draw_Block(env, geometry_object, transform, clipbox, plask_real_path)

_geometry_drawers[plask.geometry.Cylinder] = draw_Cylinder


def draw_Prism(env, geometry_object, transform, clipbox, plask_real_path):
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

        draw_Block(env, geometry_object, transform, clipbox, plask_real_path)


_geometry_drawers[plask.geometry.Prism] = draw_Prism


def draw_Extrusion(env, geometry_object, transform, clipbox, plask_real_path):
    if env.axes == (1, 2) or env.axes == (2, 1):
        try:
            env.axes = tuple(x-1 for x in env.axes)  # change axes to 2D
            draw_geometry_object(env, geometry_object.item, transform, clipbox, plask_real_path + [0])
        except RuntimeError:
            return
        finally:    # revert axes settings, change back to 3D:
            env.axes = tuple(x+1 for x in env.axes)
    else:
        #draw_Block(env, geometry_object, transform, clipbox)  # draw block uses bbox, so it will work fine
        for leaf_bbox in geometry_object.get_leafs_bboxes():
            draw_bbox(env, None, leaf_bbox, transform, clipbox, plask_real_path)

_geometry_drawers[plask.geometry.Extrusion] = draw_Extrusion


def draw_Revolution(env, geometry_object, transform, clipbox, plask_real_path):
    if env.axes == (0, 1) or env.axes == (1, 0):    # view from the top
        try:
            obj2d = geometry_object.item
        except RuntimeError:
            return
        rads = set()
        for bb in obj2d.get_leafs_bboxes():
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
            draw_geometry_object(env, geometry_object.item, transform, clipbox, new_plask_real_path)
            draw_Flipped(env, geometry_object.item, transform, clipbox, 0, new_plask_real_path)
            #draw_Block(env, geometry_object, transform, clipbox)
        except RuntimeError:
            return
        finally:
            env.axes = original_axes

_geometry_drawers[plask.geometry.Revolution] = draw_Revolution


def draw_Translation(env, geometry_object, transform, clipbox, plask_real_path):
    new_transform = matplotlib.transforms.Affine2D()
    t = geometry_object.vec
    new_transform.translate(t[env.axes[0]], t[env.axes[1]])
    try:
        draw_geometry_object(env, geometry_object.item, new_transform + transform, clipbox, plask_real_path + [0])
    except RuntimeError:
        return


_geometry_drawers[plask.geometry.Translation2D] = draw_Translation
_geometry_drawers[plask.geometry.Translation3D] = draw_Translation


def draw_Lattice(env, geometry_object, transform, clipbox, plask_real_path):
    for index, child in enumerate(geometry_object):
        draw_geometry_object(env, child, transform, clipbox, plask_real_path + [index])
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

_geometry_drawers[plask.geometry.Lattice] = draw_Lattice


def draw_Flipped(env, geometry_object, transform, clipbox, axis_nr, plask_real_path):
    if axis_nr == env.axes[0]:
        draw_geometry_object(env, geometry_object, matplotlib.transforms.Affine2D.from_values(-1.0, 0, 0, 1.0, 0, 0) + transform, clipbox, plask_real_path)
    elif axis_nr == env.axes[1]:
        draw_geometry_object(env, geometry_object, matplotlib.transforms.Affine2D.from_values(1.0, 0, 0, -1.0, 0, 0) + transform, clipbox, plask_real_path)
    else:
        draw_geometry_object(env, geometry_object, transform, clipbox, plask_real_path)


def draw_Flip(env, geometry_object, transform, clipbox, plask_real_path):
    try:
        draw_Flipped(env, geometry_object.item, transform, clipbox, geometry_object.axis_nr, plask_real_path + [0])
    except RuntimeError:
        return

_geometry_drawers[plask.geometry.Flip2D] = draw_Flip
_geometry_drawers[plask.geometry.Flip3D] = draw_Flip


def draw_Mirror(env, geometry_object, transform, clipbox, plask_real_path):
    #TODO modify clip-box?
    try:
        draw_geometry_object(env, geometry_object.item, transform, clipbox, plask_real_path + [0])
    except RuntimeError:
        return
    if geometry_object.axis_nr in env.axes:  # in 3D this must not be true
        draw_Flip(env, geometry_object, transform, clipbox, plask_real_path)

_geometry_drawers[plask.geometry.Mirror2D] = draw_Mirror
_geometry_drawers[plask.geometry.Mirror3D] = draw_Mirror


def _b(bound):
    return math.copysign(1e100, bound) if math.isinf(bound) else bound


def draw_clipped(env, geometry_object, transform, clipbox, new_clipbox, plask_real_path):
    """Used by draw_Clip and draw_Intersection."""

    if new_clipbox.upper[env.axes[0]] < new_clipbox.lower[env.axes[0]] or \
       new_clipbox.upper[env.axes[1]] < new_clipbox.lower[env.axes[1]]:
        return

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
        draw_geometry_object(env, geometry_object, transform, clipbox, plask_real_path)
    # else, if clipbox is empty now, it will be never non-empty, so all will be clipped-out


def draw_Clip(env, geometry_object, transform, clipbox, plask_real_path):
    try:
        draw_clipped(env, geometry_object.item, transform, clipbox, geometry_object.clipbox, plask_real_path + [0])
    except RuntimeError:
        return

_geometry_drawers[plask.geometry.Clip2D] = draw_Clip
_geometry_drawers[plask.geometry.Clip3D] = draw_Clip


def draw_Intersection(env, geometry_object, transform, clipbox, plask_real_path):
    try:
        if geometry_object.envelope is not None:
            draw_clipped(env, geometry_object.item, transform, clipbox, geometry_object.envelope.bbox,
                          plask_real_path + [0])
        else:
            draw_geometry_object(env, geometry_object.item, transform, clipbox, plask_real_path + [0])
    except RuntimeError:
        return

_geometry_drawers[plask.geometry.Intersection2D] = draw_Intersection
_geometry_drawers[plask.geometry.Intersection3D] = draw_Intersection


def draw_geometry2d(env, geometry_object, transform, clipbox, plask_real_path):
    try:
        draw_geometry_object(env, geometry_object.item, transform, clipbox, plask_real_path + [0, 0])
    except RuntimeError:
        return

_geometry_drawers[plask.geometry.Cartesian2D] = draw_geometry2d
_geometry_drawers[plask.geometry.Cylindrical] = draw_geometry2d



def draw_geometry_object(env, geometry_object, transform, clipbox, plask_real_path=None):
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
                draw_geometry_object(env, child, transform, clipbox, plask_real_path + [index])
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


def plot_geometry(geometry, color=None, lw=1.0, plane=None, zorder=None, mirror=False, periods=True, fill=False,
                  axes=None, figure=None, margin=None, get_color=None, alpha=1.0, extra=None, picker=None,
                  edges=False, edge_alpha=0.25, edge_lw=None):
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

        periods (bool): If *True*, all visible periods are plotted in the periodic
                geometries.

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

        picker (None|float|boolean|callable): Matplotlib picker attribute
                for all artists appended to plot (see matplotlib doc.).

        edges (bool): If *True*, the geometry edges are plotted.

        edge_alpha (float): Opacity of edges if they are plotted.

        edge_lw (None|float): Linewidth for the edges. If *None*, it is zero for filled
                plots and equal to `lw` for wireframes.

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

    if margin is None:
        lims = axes.get_xlim(), axes.get_ylim()

    cyl = isinstance(geometry, plask.geometry.Cylindrical)

    # if isinstance(geometry, plask.geometry.Cartesian3D):
    if geometry.dims == 3:
        fill = False    # we ignore fill parameter in 3D
        dd = 0
        #if plane is None: plane = 'xy'
        ax = _get_2d_axes(plane)
        dirs = tuple((("back", "front"), ("left", "right"), ("bottom", "top"))[i] for i in ax)
    else:
        dd = 1
        ax = 0, 1
        dirs = (("inner", "outer") if cyl else ("left", "right"),
                ("bottom", "top"))

    try:
        bg = (1. - edge_alpha)  * array(to_rgb(axes.get_facecolor()))
    except AttributeError:
        bg = (1. - edge_alpha)  * array(to_rgb(axes.get_axis_bgcolor()))

    if zorder is None:
        zorder = 0.5 if fill else 2.0

    env = DrawEnviroment(ax, axes, fill, color, get_color, lw, alpha, zorder=zorder, picker=picker,
                         extra=extra)

    draw_geometry_object(env, geometry, axes.transData, None)

    env.picker = None   # below we draw only some visuals with no need to pick anything

    try:
        geometry_edges = (geometry.edges[dirs[0][0]], geometry.edges[dirs[0][1]]), \
                         (geometry.edges[dirs[1][0]], geometry.edges[dirs[1][1]])
    except AttributeError:
        geometry_edges = (None, None), (None, None)

    if edge_lw is None: edge_lw = lw
    eec = edge_alpha * array(to_rgb(env.color)) + bg

    _get_color = env.get_color
    edge_get_color = lambda m: edge_alpha * array(to_rgb(_get_color(m))) + bg

    ezo = zorder - 0.001
    epzo = zorder - 0.002

    have_extends = edges and any(geometry_edges[i][j] == 'extend' for i in (0,1) for j in (0,1))

    # Draw uniform edges
    if edges:
        try:
            bbox = geometry.bbox
            ecd = [[None, None], [None, None]]
            ecm = [[None, None], [None, None]]
            for i in 0, 1:
                lo = geometry_edges[i][0]
                hi = geometry_edges[i][1]
                if not lo or lo == 'air': lo = ''
                if not hi or hi == 'air': hi = ''
                if lo == 'periodic' or hi == 'periodic' or (lo == 'mirror' and hi == 'mirror'):
                    continue
                if lo == 'mirror':
                    if hi != 'extend':
                        ecd[i][0] = -bbox.upper[ax[i]]
                        ecd[i][1] = bbox.upper[ax[i]]
                        ecm[i][0] = ecm[i][1] = hi
                elif hi == 'mirror':
                    if lo != 'extend':
                        ecd[i][0] = bbox.lower[ax[i]]
                        ecd[i][1] = - bbox.lower[ax[i]]
                        ecm[i][0] = ecm[i][1] = lo
                else:
                    if lo != 'extend':
                        ecd[i][0], ecm[i][0] = bbox.lower[ax[i]], lo
                    if hi != 'extend':
                        ecd[i][1], ecm[i][1] = bbox.upper[ax[i]], hi
            eco = [[None, None], ecd[1]] if ax[0] < ax[1] else [ecd[0], [None, None]]
            if cyl:
                if ecm[0][0] and ecd[0][0] > 0.:
                    axes.add_patch(Plane(x0=-ecd[0][0], x1=ecd[0][0], y0=eco[1][0], y1=eco[1][1], fc=edge_get_color(ecm[0][0]),
                                         lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
                if ecm[0][1]:
                    axes.add_patch(Plane(x1=-ecd[0][1], y0=eco[1][0], y1=eco[1][1], fc=edge_get_color(ecm[0][1]),
                                         lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
                    axes.add_patch(Plane(x0=ecd[0][1], y0=eco[1][0], y1=eco[1][1], fc=edge_get_color(ecm[0][1]),
                                         lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
            else:
                if ecm[0][0]:
                    axes.add_patch(Plane(x1=ecd[0][0], y0=eco[1][0], y1=eco[1][1], fc=edge_get_color(ecm[0][0]),
                                         lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
                if ecm[0][1]:
                    axes.add_patch(Plane(x0=ecd[0][1], y0=eco[1][0], y1=eco[1][1], fc=edge_get_color(ecm[0][1]),
                                         lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
            if ecm[1][0]:
                axes.add_patch(Plane(y1=ecd[1][0], x0=eco[0][0], x1=eco[0][1], fc=edge_get_color(ecm[1][0]),
                                     lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
            if ecm[1][1]:
                axes.add_patch(Plane(y0=ecd[1][1], x0=eco[0][0], x1=eco[0][1], fc=edge_get_color(ecm[1][1]),
                                     lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
        except (AttributeError, KeyError):
            pass

    hmirror = False
    vmirror = False

    if mirror or periods or edges:
        bbox = geometry.bbox
        (left, right), (bottom, top) = ((bbox.lower[a], bbox.upper[a]) for a in ax)
        hshift, vshift = (bbox.size[a] for a in ax)
        try:
            if geometry_edges[0][0] == 'mirror' or geometry_edges[0][1] == 'mirror' or cyl:
                if (cyl and left < 0.) or \
                   (geometry_edges[0][0] == 'mirror' and left != 0.) or \
                   (geometry_edges[0][1] == 'mirror' and right != 0.):
                    plask.print_log('warning', 'Mirror is not located at the axis')
                    hmirror = False
                else:
                    hshift *= 2
                    hmirrortransform = matplotlib.transforms.Affine2D.from_values(-1., 0, 0, 1., 0, 0)
                    hmirror = True
            else:
                hmirror = False
            if geometry_edges[1][0] == 'mirror' or geometry_edges[1][1] == 'mirror':
                if (geometry_edges[1][0] == 'mirror' and bottom != 0.) or \
                   (geometry_edges[1][1] == 'mirror' and top != 0.):
                    plask.print_log('warning', 'Mirror is not located at the axis')
                    vmirror = False
                else:
                    vshift *= 2
                    vmirrortransform = matplotlib.transforms.Affine2D.from_values(1., 0, 0, -1., 0, 0)
                    vmirror = True
                    if hmirror:
                        vhmirrortransform = matplotlib.transforms.Affine2D.from_values(-1., 0, 0, -1., 0, 0)
            else:
                vmirror = False
            if not (geometry_edges[0][0] == 'periodic' and (geometry_edges[0][1] in ('periodic', 'mirror')) or
                    geometry_edges[0][0] == 'mirror' and geometry_edges[0][1] == 'periodic'):
                hshift = 0
            if not (geometry_edges[1][0] == 'periodic' and (geometry_edges[1][1] in ('periodic', 'mirror')) or
                    geometry_edges[1][0] == 'mirror' and geometry_edges[1][1] == 'periodic'):
                vshift = 0
        except AttributeError:  # we draw non-Geometry object
            pass

        color = env.color

        def _set_env_style(cond, per=False):
            if cond:
                env.lw = lw
                env.get_color = _get_color
                env.color = color
                env.zorder = zorder
            else:
                env.lw = edge_lw
                env.get_color = edge_get_color
                env.color = eec
                env.zorder = ezo if not per else epzo

        if hmirror or vmirror:
            if hmirror:
                _set_env_style(mirror or (periods and hshift))
                draw_geometry_object(env, geometry, hmirrortransform + axes.transData, None)
            if vmirror:
                _set_env_style(mirror or (periods and vshift))
                draw_geometry_object(env, geometry, vmirrortransform + axes.transData, None)
                if hmirror:
                    _set_env_style(mirror or (periods and hshift and vshift))
                    draw_geometry_object(env, geometry, vhmirrortransform + axes.transData, None)

        if hshift or vshift:
            env.periodic = {'dx': hshift, 'dy': vshift}
            _set_env_style(periods, True)
            draw_geometry_object(env, geometry, axes.transData, None)
            if hmirror:
                _set_env_style(periods and (mirror or hshift), True)
                draw_geometry_object(env, geometry, hmirrortransform + axes.transData, None)
            if vmirror:
                _set_env_style(periods and (mirror or vshift), True)
                draw_geometry_object(env, geometry, vmirrortransform + axes.transData, None)
                if hmirror:
                    _set_env_style(periods and (mirror or (hshift and vshift)), True)
                    draw_geometry_object(env, geometry, vhmirrortransform + axes.transData, None)
            env.periodic = None

        hvmirror = hmirror, vmirror

        if have_extends:
            if geometry.dims == 3:
                geometry_mesh = plask.mesh.Rectangular3D.SimpleGenerator()(geometry)
                v = ({0, 1, 2} - set(ax)).pop()
                vmesh = getattr(geometry_mesh, "axis{:1d}".format(v)).midpoints
                try:
                    nair = len(vmesh) * [plask.material.get('air')]
                except ValueError:
                    nair = len(vmesh) * [None]
            else:  # geometry.dims == 2
                geometry_mesh = plask.mesh.Rectangular2D.SimpleGenerator()(geometry)
            emesh = [getattr(geometry_mesh, "axis{:1d}".format(a)) for a in ax]
            blu = bbox.lower, bbox.upper
            for i,a in enumerate(ax):
                for j in (0,1):
                    if geometry_edges[i][j] != 'extend': continue
                    e = blu[j][a]
                    xy = [[None, None], [None, None]]
                    xy[i][1-j] = e
                    xym = [[None, None], [None, None]]
                    xym[i][j] = -e
                    em = emesh[1-i]

                    def _add_extend(g0, g1):
                        if cyl and i == 0:
                            if j == 0:
                                if e != 0:
                                    axes.add_patch(Plane(x0=-e, x1=e, y0=g0, y1=g1, fc=fc,
                                                        lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
                            else:
                                axes.add_patch(Plane(x1=-e, y0=g0, y1=g1, fc=fc, lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
                                axes.add_patch(Plane(x0=+e, y0=g0, y1=g1, fc=fc, lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
                        else:
                            xy[1-i] = g0, g1
                            axes.add_patch(Plane(x0=xy[0][0], x1=xy[0][1], y0=xy[1][0], y1=xy[1][1], fc=fc,
                                                lw=edge_lw, fill=fill, ec=eec, zorder=ezo))
                            if hvmirror[i]:
                                xym[1-i] = g0, g1
                                axes.add_patch(Plane(x0=xym[0][0], x1=xym[0][1], y0=xym[1][0], y1=xym[1][1], fc=fc,
                                                    lw=edge_lw, fill=fill, ec=eec, zorder=ezo))

                    fc = None
                    pm = None
                    g0 = g1 = em[0]
                    ng = len(em)
                    am = True
                    for g2 in em[1:]:
                        if geometry.dims == 2:
                            p = [None, None]
                            p[a] = e
                            p[1-a] = 0.5 * (g1 + g2)
                            m = geometry.get_material(p)
                            if pm is not None:
                                fc = edge_get_color(pm)
                            am = str(pm) != 'air'
                        else:
                            m = [geometry.get_material(**{
                                    "c{:1d}".format(a): e,
                                    "c{:1d}".format(ax[1-i]): 0.5 * (g1 + g2),
                                    "c{:1d}".format(v): cv,
                                }) for cv in vmesh]
                            am = pm != nair
                        if m != pm:
                            if g0 != g1 and am:
                                _add_extend(g0, g1)
                                if hvmirror[1-i]:
                                    _add_extend(-g1, -g0)
                            g0 = g1
                        g1 = g2
                        pm = m
                    if geometry.dims == 2:
                        am = str(m) != 'air'
                        fc = edge_get_color(m)
                    else:
                        am = m != nair
                    if am:
                        _add_extend(g0, g1)
                        if hvmirror[1-i]:
                            _add_extend(-g1, -g0)

    if margin is not None:
        box = geometry.bbox
        if mirror and hmirror:
            m = max(abs(box.lower[ax[0]]), abs(box.upper[ax[0]]))
            m += 2. * m * margin
            axes.set_xlim(-m, m)
        else:
            m = (box.upper[ax[0]] - box.lower[ax[0]]) * margin
            axes.set_xlim(box.lower[ax[0]] - m, box.upper[ax[0]] + m)
        if mirror and vmirror:
            m = max(abs(box.lower[ax[1]]), abs(box.upper[ax[1]]))
            m += 2. * m * margin
            axes.set_ylim(-m, m)
        else:
            m = (box.upper[ax[1]] - box.lower[ax[1]]) * margin
            axes.set_ylim(box.lower[ax[1]] - m, box.upper[ax[1]] + m)
    else:
        axes.set_xlim(*lims[0])
        axes.set_ylim(*lims[1])

    if ax[0] > ax[1] and not axes.yaxis_inverted():
        axes.invert_yaxis()

    axes.set_xlabel(u"${}$ [µm]".format(plask.config.axes[dd+ax[0]]))
    axes.set_ylabel(u"${}$ [µm]".format(plask.config.axes[dd+ax[1]]))

    if extra is not None:
        return axes, env.extra_patches
    else:
        return axes
