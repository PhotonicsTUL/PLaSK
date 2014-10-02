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


class GNTransform(GNObject):

    def children_from_XML(self, ordered_reader, conf):
        construct_geometry_object(ordered_reader.require(), conf)


class GNTranslation(GNTransform):

    def __init__(self, parent = None, dim = None):
        super(GNObject, self).__init__(parent=parent, dim=dim, children_dim=dim)

    def attributes_from_XML(self, attribute_reader, conf):
        pass    #TODO

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNTranslation(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNTranslation(dim = 3)
        result.set_XML_element(element, conf)
        return result


class GNClip(GNTransform):

    def __init__(self, parent = None, dim = None):
        super(GNObject, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.left = None
        self.right = None
        self.bottom = None
        self.top = None
        if dim == 3:
            self.back = None
            self.front = None

    def attributes_from_XML(self, attribute_reader, conf):
        self.left = conf.get('left')
        self.right = conf.get('right')
        self.bottom = conf.get('bottom')
        self.top = conf.get('top')
        if self.dim == 3:
            self.back = conf.get('back')
            self.front = conf.get('front')

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNClip(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNClip(dim = 3)
        result.set_XML_element(element, conf)
        return result