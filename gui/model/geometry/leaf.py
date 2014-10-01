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
        
    def attributes_from_XML(self, attribute_reader, conf):
        super(GNLeaf, self).attributes_from_XML(attribute_reader, conf)
        self.step_num = attribute_reader.get('step-num')
        self.step_dist = attribute_reader.get('step-dist')
        #TODO material
        

class GNBlock(GNLeaf):

    def __init__(self, parent = None, dim = None):
        super(GNBlock, self).__init__(parent=parent, dim=dim)

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNBlock, self).attributes_from_XML(attribute_reader, conf)
        #TODO size

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNBlock(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNBlock(dim = 3)
        result.set_XML_element(element, conf)
        return result


class GNCylinder(GNLeaf):
    
    def __init__(self, parent = None):
        super(GNCylinder, self).__init__(parent=parent, dim = 3)
        self.radius = None  #required in PLaSK but not in GUI
        self.height = None  #required in PLaSK but not in GUI

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNCylinder, self).attributes_from_XML(attribute_reader, conf)
        self.radius = attribute_reader.get('radius')
        self.height = attribute_reader.get('height')

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNCylinder()
        result.set_XML_element(element, conf)
        return result
    

class GNCircle(GNLeaf):
    
    def __init__(self, parent = None, dim = None):
        super(GNCircle, self).__init__(parent=parent, dim=dim)
        self.radius = None  #required in PLaSK but not in GUI

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNCircle, self).attributes_from_XML(attribute_reader, conf)
        self.radius = attribute_reader.get('radius')

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNCircle(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNCircle(dim = 3)
        result.set_XML_element(element, conf)
        return result


class GNTriangle(GNLeaf):

    def __init__(self, parent = None):
        super(GNTriangle, self).__init__(parent=parent)
        #TODO points

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNTriangle, self).attributes_from_XML(attribute_reader, conf)
        #TODO points

    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNTriangle()
        result.set_XML_element(element, conf)
        return result
