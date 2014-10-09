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

from .node import GNode


class GNObject(GNode):
    '''Base class for all nodes read by GeometryReader::readObject() in PLaSK.'''

    def __init__(self, parent = None, dim = None, children_dim = None):
        super(GNObject, self).__init__(parent, dim, children_dim)
        self.name = None
        self.role = None
        self.axes = None

    def attributes_from_XML(self, attribute_reader, conf):
        self.name = attribute_reader.get('name')
        self.role = attribute_reader.get('role')
        self.axes = attribute_reader.get('axes')
        conf.axes = self.axes

