# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

try:
    from plask import axeslist_by_name
except ImportError:
    pass


try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str


class GNAligner(object):

    POSITION_LOW = 0
    POSITION_CENTER = 1
    POSITION_HIGH = 2
    POSITION_ORIGIN = 3

    def __init__(self, position=None, value=None):
        super(GNAligner, self).__init__()
        self.position = position
        self.value = value

    @staticmethod
    def names(dims, axis_names_in_dims, axis_nr):
        """
        Get name of aligners.
        :param int dims: number of dims, 2 or 3
        :param list axis_names_in_dims: names of axis in dims (list of length dim)
        :param int axis_nr: axis number, from 0 to dims-1
        :return: tuple with aligner names: lo, center, hi, origin, center (alternative name)
        """
        a = axis_names_in_dims[axis_nr]
        return (('back', a + 'center', 'front', a, 'longcenter'),
                ('left', a + 'center', 'right', a, 'trancenter'),
                ('bottom', a + 'center', 'top', a, 'vertcenter'))[(axis_nr + 1) if dims == 2 else axis_nr]

    @staticmethod
    def display_names(dims, axis_nr):
        """
        Get name of aligners.
        :param int dims: number of dims, 2 or 3
        :param int axis_nr: axis number, from 0 to dims-1
        :return: tuple with aligner names: lo, center, hi, origin, center (alternative name)
        """
        return (('back', 'center', 'front', 'origin'),
                ('left', 'center', 'right', 'origin'),
                ('bottom', 'center', 'top', 'origin'))[(axis_nr + 1) if dims == 2 else axis_nr]

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, basestring):
            if not value:
                self._position = None
            elif 'center' in value:
                self._position = GNAligner.POSITION_CENTER
            elif value in ('back', 'left', 'bottom'):
                self._position = GNAligner.POSITION_LOW
            elif value in ('front', 'right', 'top'):
                self._position = GNAligner.POSITION_HIGH
            else:
                self._position = GNAligner.POSITION_ORIGIN
        else:
            self._position = value

    def position_str(self, dims, axis_names_in_dims, axis_nr):
        return None if self.position is None else GNAligner.names(dims, axis_names_in_dims, axis_nr)[self.position]


def axes_as_list(name_or_list):
    """
    Convert a string to axes list.
    :param str name_or_list: string which describe axes SCHEME (can also be a list)
    :return: list of 3 axes names, described by name_or_list string or name_or_list if it is a list or
                neutral axes names if name_or_list is improper string
    """
    try:
        return axeslist_by_name(name_or_list.encode('utf-8')) if isinstance(name_or_list, basestring) \
               else list(name_or_list)
    except:
        return ['long', 'tran', 'vert']


def axes_dim(axes, dim):
    """
    Calculate axes names for given number of dimensions.
    :param list axes: 3D axes
    :param int dim: number of dimensions, 2 or 3
    :return list: list with axes names
    """
    return axes_as_list(axes) if dim == 3 else axes_as_list(axes)[1:]


class GNReadConf(object):
    """ Configuration using while geometry objects are read.
        Stores information about expected suffix, axes configuration and parent node for new elements.
    """

    def __init__(self, conf=None, axes=None, parent=None):
        super(GNReadConf, self).__init__()
        self._axes = ['long', 'tran', 'vert']
        if conf is not None:
            self.axes = conf.axes
            self.parent = conf.parent
            self.object_names = conf.object_names
        else:
            self.parent = None
            self.object_names = dict()
        if axes is not None: self.axes = axes
        if parent is not None: self.parent = parent

    @property
    def dim(self):
        return None if self.parent is None else self.parent.children_dim

    @property
    def suffix(self):
        d = self.dim
        return None if d is None else '{}d'.format(d)

    def axes_names(self, dim=None):
        return axes_dim(self.axes, self.parent.dim if dim is None else dim)

    def axis_name(self, dim, axis_nr):
        return self.axes_names(dim)[axis_nr]

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, name_or_list):
        if name_or_list is None: return
        self._axes = axes_as_list(name_or_list)

    def aligners(self, dims, axis_nr):
        if dims is None: dims = self.parent.dim
        return GNAligner.names(dims, self.axes_names(dims), axis_nr)

    def read_aligners(self, attribute_reader, dims=None, *axes_to_read):
        """
        Read aligners.
        :param attribute_reader:
        :param dims:
        :param axes_to_read: numbers of axes to read, default to range(0, dims) (if not given any)
        :return: list of GNAligner for each axis to read (GNAligner with None-s if aligner for given axis was not read)
        """
        if dims is None: dims = self.parent.dim
        to_read = range(0, dims) if not axes_to_read else axes_to_read
        res = [GNAligner(None, None) for _ in to_read]
        for axis_nr in to_read:
            for position_name in self.aligners(dims, axis_nr):
                value = attribute_reader.get(position_name)
                if value is not None:
                    res[axis_nr].position = position_name
                    res[axis_nr].value = value
                    break
        return res

    def write_aligners(self, element, dims=None, aligners={}):
        if isinstance(aligners, (list, tuple)):
            if dims is None: dims = len(aligners)
            aligners = {axis_nr: aligners[axis_nr] for axis_nr in range(0, dims)}
        else:
            if dims is None: dims = self.parent.dim
        for axis_nr, aligner in aligners.items():
            pos_str = aligner.position_str(dims, self.axes_names(dims), axis_nr)
            if pos_str is not None and aligner.value:
                element.attrib[pos_str] = aligner.value

    def find_object_by_name(self, name):
        return self.object_names.get(name)

    def after_read(self, node):
        name = getattr(node, 'name', None)
        if name is not None: self.object_names[name] = node


def axes_to_str(axes, none_axes_result=''):
    if axes is None: return none_axes_result
    if isinstance(axes, basestring): return axes
    return ', '.join(axes)