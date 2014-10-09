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


class GNReadConf(object):
    """ Configuration using while geometry objects are read.
        Stores information about expected suffix, axes configuration and parent node for new elements.
    """

    def __init__(self, conf=None, axes=None, parent=None):
        super(GNReadConf, self).__init__()
        if conf is not None:
            self.axes = conf.axes
            self.parent = conf.parent
        else:
            self.axes = ['z', 'x', 'y']
            self.parent = None
        if axes is not None: self.axes = axes
        if parent is not None: self.parent = parent

    @property
    def dim(self):
        return None if self.parent is None else self.parent.children_dim

    @property
    def suffix(self):
        d = self.dim
        return None if d is None else '{}d'.format(d)

    def axes_names(self, dim = None):
        if dim == None: dim = self.parent.dim
        return self.axes if dim == 3 else self.axes[1:]

    def axis_name(self, dim, axes_nr):
        return self.axes_dim(dim)[axes_nr]

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, name_or_list):
        if name_or_list is None: return
        self._axes =  axeslist_by_name(axes) if isinstance(name_or_list, basestring) else list(name_or_list)