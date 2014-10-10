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
from .node import GNode
from ...utils.xml import AttributeReader, OrderedTagReader
from . import GNAligner

class GNZero(GNode):

    def __init__(self, parent = None, dim = None):
        super(GNZero, self).__init__(parent=parent, dim=dim)


class GNStack(GNObject):
    """2D/3D (multi-)stack"""

    def __init__(self, parent = None, dim = None):
        super(GNStack, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.repeat = None
        self.shift = None
        self.aligner = GNAligner(None, None)

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNStack, self).attributes_from_xml(attribute_reader, conf)
        self.repeat = attribute_reader.get('repeat')
        self.shift = attribute_reader.get('shift')
        self.aligner, = conf.read_aligners(attribute_reader, self.children_dim, self.children_dim-2)  #direction tran

    def children_from_xml(self, ordered_reader, conf):
        for c in ordered_reader.iter():
            if c.tag == 'item':
                with OrderedTagReader(c) as item_child_reader:
                    child = construct_geometry_object(item_child_reader.require(), conf)
                with AttributeReader(c) as item_attr_reader:
                    child.in_parent, = conf.read_aligners(item_attr_reader, self.children_dim, self.children_dim-2)  #direction tran
            elif c.tag == 'zero':
                GNZero(self, self.children_dim)
            else:
                construct_geometry_object(c, conf)

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNStack(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNStack(dim = 3)
        result.set_xml_element(element, conf)
        return result


#TODO GNShelf as separate class or support shelf by GNStack


class GNAlignContainer(GNObject):
    """2D/3D align container"""

    def __init__(self, parent = None, dim = None):
        super(GNAlignContainer, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.aligners = [GNAligner(None, None) for _ in range(0, self.children_dim)]

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNAlignContainer, self).attributes_from_xml(attribute_reader, conf)
        self.aligners = conf.read_aligners(attribute_reader, self.children_dim)

    def children_from_xml(self, ordered_reader, conf):
        for c in ordered_reader.iter():
            if c.tag == 'item':
                with OrderedTagReader(c) as item_child_reader:
                    child = construct_geometry_object(item_child_reader.require(), conf)
                with AttributeReader(c) as item_attr_reader:
                    child.in_parent = conf.read_aligners(item_attr_reader, self.children_dim)
            else:
                construct_geometry_object(c, conf)

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNAlignContainer(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNAlignContainer(dim = 3)
        result.set_xml_element(element, conf)
        return result