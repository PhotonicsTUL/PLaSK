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
from ...utils.xml import xml_to_attr


class GNTransform(GNObject):

    def children_from_XML(self, ordered_reader, conf):
        construct_geometry_object(ordered_reader.get(), conf)


class GNTranslation(GNTransform):

    def __init__(self, parent = None, dim = None):
        super(GNTranslation, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.pos = [None for _ in range(0, dim)]

    def attributes_from_XML(self, attribute_reader, conf):
        self.pos = [attribute_reader.get(a) for a in conf.axes_names(self.dim)]

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
        super(GNClip, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.left = None
        self.right = None
        self.bottom = None
        self.top = None
        if dim == 3:
            self.back = None
            self.front = None

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNClip, self).attributes_from_XML(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'left', 'right', 'bottom', 'top')
        if self.dim == 3:
            xml_to_attr(attribute_reader, self, 'back', 'front')


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


class GNFlip(GNTransform):

    def __init__(self, parent = None, dim = None):
        super(GNFlip, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.axis = None

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNFlip, self).attributes_from_XML(attribute_reader, conf)
        self.axis = attribute_reader.get('axis')

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNFlip(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNFlip(dim = 3)
        result.set_XML_element(element, conf)
        return result


class GNMirror(GNTransform):

    def __init__(self, parent = None, dim = None):
        super(GNMirror, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.axis = None

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNMirror, self).attributes_from_XML(attribute_reader, conf)
        self.axis = attribute_reader.get('axis')

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNMirror(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNMirror(dim = 3)
        result.set_XML_element(element, conf)
        return result


class GNExtrusion(GNTransform):

    def __init__(self, parent = None):
        super(GNExtrusion, self).__init__(parent=parent, dim=3, children_dim=2)
        self.length = None
        
    def attributes_from_XML(self, attribute_reader, conf):
        super(GNExtrusion, self).attributes_from_XML(attribute_reader, conf)
        self.length = attribute_reader.get('length')

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNExtrusion()
        result.set_XML_element(element, conf)
        return result


class GNRevolution(GNTransform):

    def __init__(self, parent = None):
        super(GNExtrusion, self).__init__(parent=parent, dim=3, children_dim=2)

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNRevolution()
        result.set_XML_element(element, conf)
        return result