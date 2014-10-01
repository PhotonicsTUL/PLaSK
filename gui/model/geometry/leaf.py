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

class GNBlock(GNObject):

    def __init__(self, parent = None, dim = None):
        super(GNBlock, self).__init__(parent=parent, dim=dim)

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
