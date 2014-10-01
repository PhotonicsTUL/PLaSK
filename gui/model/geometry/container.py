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

from .object import GNObject
from .types import construct_geometry_object

class GNStack(GNObject):
    """2D/3D (multi-)stack"""

    def __init__(self, parent = None, dim = None):
        super(GNStack, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.repeat = '1'
        self.shift = '0'

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNStack, self).attributes_from_XML(attribute_reader, conf)
        self.repeat = attribute_reader.get('repeat', '1')
        self.shift = attribute_reader.get('shift', '0')
        #TODO default aligners

    def children_from_XML(self, ordered_reader, conf):
        for c in ordered_reader.iter():
            if c.tag == 'item':
                pass    #TODO!!
            elif c.tag == 'zero':
                pass    #TODO!!
            else:
                construct_geometry_object(c, conf)

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNStack(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNStack(dim = 3)
        result.set_XML_element(element, conf)
        return result


class GNAlignContainer(GNObject):
    """2D/3D align container"""

    def __init__(self, parent = None, dim = None):
        super(GNAlignContainer, self).__init__(parent=parent, dim=dim, children_dim=dim)

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNAlignContainer, self).attributes_from_XML(attribute_reader, conf)
        #TODO default aligners

    def children_from_XML(self, ordered_reader, conf):
        for c in ordered_reader.iter():
            if c.tag == 'item':
                pass    #TODO!!
            else:
                construct_geometry_object(c, conf)

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNAlignContainer(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNAlignContainer(dim = 3)
        result.set_XML_element(element, conf)
        return result