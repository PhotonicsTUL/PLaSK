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
        Stores information about expected suffix and axes configuration.
    """

    def __init__(self, conf=None, suffix=None, axes=None):
        super(GNReadConf, self).__init__()
        if conf is not None:
            self.suffix = conf.suffix
            self.axes = conf.axes
        else:
            self.suffix = None
            self.axes = None
        if suffix is not None: self.suffix = suffix
        if axes is not None: self.axes = axes

