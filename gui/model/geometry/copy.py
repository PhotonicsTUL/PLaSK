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
from .object import GNObject


class GNAgain(GNode):

    def __init__(self, parent = None, ref = None):
        super(GNAgain, self).__init__(parent)
        self.ref = ref

    def attributes_from_XML(self, attribute_reader):
        self.ref = attribute_reader.require('ref')



# ---------- copy tag and its children: --------------

class GNCopyChild(GNode):

    def __init__(self, parent = None, object = None):
        super(GNCopyChild, self).__init__(parent)
        self.object = object

    def attributes_from_XML(self, attribute_reader):
        self.object = attribute_reader.require('object')



class GNCDelete(GNCopyChild):
    pass


class GNCReplace(GNCopyChild):

    def __init__(self, parent = None, object = None, replacer = None):
        super(GNCReplace, self).__init__(parent, object)
        self.replacer = replacer    # with in PLaSK

    def attributes_from_XML(self, attribute_reader):
        super(GNCReplace, self).attributes_from_XML(attribute_reader)
        self.replacer = attribute_reader.get('with', None)

    def children_from_XML(self, ordered_reader):
        if self.replacer is not None:
            pass    #TODO read geometry object


class GNCToBlock(GNCopyChild):

    def __init__(self, parent = None, object = None, material = None):
        super(GNCToBlock, self).__init__(parent, object)
        self.material = material    # with in PLaSK

    def attributes_from_XML(self, attribute_reader):
        super(GNCToBlock, self).attributes_from_XML(attribute_reader)
        self.material = attribute_reader.get('material', None)


class GNCopy(GNObject):

    def __init__(self, parent = None, name = None, source = None):
        super(GNCopy, self).__init__(parent, name)
        self.source = source    # from in PLaSK

    def attributes_from_XML(self, attribute_reader):
        super(GNCopy, self).attributes_from_XML(attribute_reader)
        self.source = attribute_reader.require('from')

    #TODO def children_from_XML(self, ordered_reader):