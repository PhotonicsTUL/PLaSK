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

class GNLeaf(GNObject):

    def __init__(self, parent = None, dim = None):
        super(GNObject, self).__init__(parent=parent, dim=dim, children_dim=None)
        self.step_num = None
        self.step_dist = None
        #TODO material
        
    def attributes_from_xml(self, attribute_reader, conf):
        super(GNLeaf, self).attributes_from_xml(attribute_reader, conf)
        self.step_num = attribute_reader.get('step-num')
        self.step_dist = attribute_reader.get('step-dist')
        #TODO material (also top/bottom)
        

class GNBlock(GNLeaf):

    def __init__(self, parent = None, dim = None):
        super(GNBlock, self).__init__(parent=parent, dim=dim)
        self.size = [None for _ in range(0, dim)]

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNBlock, self).attributes_from_xml(attribute_reader, conf)
        self.pos = [attribute_reader.get(a) for a in conf.axes_names(self.dim)]

    def tag_name(self, full_name = True):
        return "block{}d".format(self.dim) if full_name else "block"

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNBlock(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNBlock(dim = 3)
        result.set_xml_element(element, conf)
        return result


class GNCylinder(GNLeaf):
    
    def __init__(self, parent = None):
        super(GNCylinder, self).__init__(parent=parent, dim = 3)
        self.radius = None  #required in PLaSK but not in GUI
        self.height = None  #required in PLaSK but not in GUI

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNCylinder, self).attributes_from_xml(attribute_reader, conf)
        self.radius = attribute_reader.get('radius')
        self.height = attribute_reader.get('height')

    def tag_name(self, full_name = True):
        return "cylinder"

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNCylinder()
        result.set_xml_element(element, conf)
        return result
    

class GNCircle(GNLeaf):
    
    def __init__(self, parent = None, dim = None):
        super(GNCircle, self).__init__(parent=parent, dim=dim)
        self.radius = None  #required in PLaSK but not in GUI

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNCircle, self).attributes_from_xml(attribute_reader, conf)
        self.radius = attribute_reader.get('radius')

    def tag_name(self, full_name = True):
        return "circle{}d".format(self.dim) if full_name else "circle"

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNCircle(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNCircle(dim = 3)
        result.set_xml_element(element, conf)
        return result


class GNTriangle(GNLeaf):

    def __init__(self, parent = None):
        super(GNTriangle, self).__init__(parent=parent, dim=2)
        self.points = ((None, None), (None, None))

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNTriangle, self).attributes_from_xml(attribute_reader, conf)
        n = conf.axes_names(2)
        r = attribute_reader
        self.points = ((r.get(n[0] + '0'), r.get(n[1] + '0')), (r.get(n[0] + '1'), r.get(n[1] + '1')))

    def tag_name(self, full_name = True):
        return "triangle"

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNTriangle()
        result.set_xml_element(element, conf)
        return result
