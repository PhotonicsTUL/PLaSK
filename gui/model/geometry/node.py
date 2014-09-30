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
from model.geometry import GNReadConf

from ...utils.xml import AttributeReader, OrderedTagReader


class GNode(object):

    def __init__(self, parent = None, children = None):
        super(GNode, self).__init__()
        self._parent = None  #used by parent property
        self.parent = parent
        self.children = [] if children is None else children
        for c in self.children: c.parent = self

    def attributes_from_XML(self, attribute_reader, conf):
        """

        :param reader:
        :param AttributeReader attribute_reader: source of attributes
        :return:
        """
        pass

    def children_from_XML(self, ordered_reader, conf):
        pass

    def from_XML(self, element, conf):
        subtree_conf = GNReadConf(conf)
        with AttributeReader(element) as a: self.attributes_from_XML(a, subtree_conf)
        with OrderedTagReader(element) as r: self.children_from_XML(r, subtree_conf)

    #def append(self, child):
    #    self.children.append(child)
    #    child.parent = self

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self.parent == parent: return
        if self.parent is not None:
            self.parent.children.remove(self)
        self.parent = parent
        parent.children.append(self)

