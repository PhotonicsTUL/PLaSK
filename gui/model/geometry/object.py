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

from ...utils.xml import AttributeReader, OrderedTagReader

class GeometryObjectModel(object):

    def __init__(self, name = None):
        super(GeometryObjectModel, self).__init__()
        self.name = name

    def attributes_from_XML(self, attribute_reader):
        """
        :param AttributeReader attribute_reader: source of attributes
        :return:
        """
        self.name = attribute_reader.get('name', None)

    def children_from_XML(self, ordered_reader):
        pass

    def from_XML(self, element):
        with AttributeReader(element) as a: self.attributes_from_XML(a)
        with OrderedTagReader(element) as r: self.children_from_XML(r)

