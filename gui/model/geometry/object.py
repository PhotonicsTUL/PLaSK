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

    def __init__(self, parent = None, name = None, role = None, step_num = None, step_dist = None):
        super(GNObject, self).__init__(parent)
        self.name = name
        self.role = role
        self.step_num = step_num
        self.step_dist = step_dist

    def attributes_from_XML(self, attribute_reader):
        """
        :param AttributeReader attribute_reader: source of attributes
        :return:
        """
        self.name = attribute_reader.get('name', None)
        self.role = attribute_reader.get('role', None)
        self.step_num = attribute_reader.get('step-num', None)
        self.step_dist = attribute_reader.get('step-dist', None)


