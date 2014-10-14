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
from ...utils.xml import xml_to_attr, attr_to_xml


class GNTransform(GNObject):

    def children_from_xml(self, ordered_reader, conf):
        construct_geometry_object(ordered_reader.get(), conf)


class GNTranslation(GNTransform):

    def __init__(self, parent = None, dim = None):
        super(GNTranslation, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.size = [None for _ in range(0, dim)]

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNTranslation, self).attributes_from_xml(attribute_reader, conf)
        axes_names = conf.axes_names(self.dim)
        alternative_names = ('depth', 'width', 'height')[(3-self.dim):]
        self.size = [None for _ in range(0, self.dim)]
        for i in range(0, self.dim):
            self.size[i] = attribute_reader.get('d' + axes_names[i])
            if self.size[i] is None:
                self.size[i] = attribute_reader.get(alternative_names[i])
                
    def attributes_to_xml(self, element, conf):
        super(GNTranslation, self).attributes_to_xml(element, conf)
        axes_names = conf.axes_names(self.dim)
        for i in range(0, self.dim):
            v = self.size[i]
            if v is not None: element.attrib['d' + axes_names[i]] = v

    def tag_name(self, full_name = True):
        return "translation{}d".format(self.dim) if full_name else "translation"

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNTranslation(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNTranslation(dim = 3)
        result.set_xml_element(element, conf)
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

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNClip, self).attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'left', 'right', 'bottom', 'top')
        if self.dim == 3:
            xml_to_attr(attribute_reader, self, 'back', 'front')

    def attributes_to_xml(self, element, conf):
        super(GNClip, self).attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'left', 'right', 'bottom', 'top')
        if self.dim == 3:
            attr_to_xml(self, element, 'back', 'front')

    def tag_name(self, full_name = True):
        return "clip{}d".format(self.dim) if full_name else "clip"

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNClip(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNClip(dim = 3)
        result.set_xml_element(element, conf)
        return result


class GNFlip(GNTransform):

    def __init__(self, parent = None, dim = None):
        super(GNFlip, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.axis = None

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNFlip, self).attributes_from_xml(attribute_reader, conf)
        self.axis = attribute_reader.get('axis')

    def attributes_to_xml(self, element, conf):
        super(GNFlip, self).attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'axis')

    def tag_name(self, full_name = True):
        return "flip{}d".format(self.dim) if full_name else "flip"

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNFlip(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNFlip(dim = 3)
        result.set_xml_element(element, conf)
        return result


class GNMirror(GNTransform):

    def __init__(self, parent = None, dim = None):
        super(GNMirror, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.axis = None

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNMirror, self).attributes_from_xml(attribute_reader, conf)
        self.axis = attribute_reader.get('axis')

    def attributes_to_xml(self, element, conf):
        super(GNMirror, self).attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'axis')

    def tag_name(self, full_name = True):
        return "mirror{}d".format(self.dim) if full_name else "mirror"

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNMirror(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNMirror(dim = 3)
        result.set_xml_element(element, conf)
        return result


class GNExtrusion(GNTransform):

    def __init__(self, parent = None):
        super(GNExtrusion, self).__init__(parent=parent, dim=3, children_dim=2)
        self.length = None
        
    def attributes_from_xml(self, attribute_reader, conf):
        super(GNExtrusion, self).attributes_from_xml(attribute_reader, conf)
        self.length = attribute_reader.get('length')

    def attributes_to_xml(self, element, conf):
        super(GNExtrusion, self).attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'length')

    def tag_name(self, full_name = True):
        return "extrusion"

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNExtrusion()
        result.set_xml_element(element, conf)
        return result


class GNRevolution(GNTransform):

    def __init__(self, parent = None):
        super(GNExtrusion, self).__init__(parent=parent, dim=3, children_dim=2)

    def tag_name(self, full_name = True):
        return "revolution"

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNRevolution()
        result.set_xml_element(element, conf)
        return result