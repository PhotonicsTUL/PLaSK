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
from lxml import etree

from .reader import GNReadConf
from ...utils.xml import AttributeReader, OrderedTagReader


class GNode(object):

    """
        :param parent: parent node of self (self will be added to parent's children)
        :param dim: number of dimension of self or None if it is unknown or not defined (like in case of again or copy)
        :param children_dim: required number of dimension of self's children or None if no or any children are allowed
    """
    def __init__(self, parent = None, dim = None, children_dim = None):
        super(GNode, self).__init__()
        self.dim = dim
        self.children_dim = children_dim
        self.children = []
        self.in_parent = None   #configuration inside parent (container)
        self._parent = None  #used by parent property
        self.parent = parent

    def attributes_from_xml(self, attribute_reader, conf):
        """

        :param reader:
        :param AttributeReader attribute_reader: source of attributes
        :return:
        """
        pass

    def children_from_xml(self, ordered_reader, conf):
        pass

    def preset_conf(self, conf):
        conf.parent = self

    def set_xml_element(self, element, conf = None):
        if conf is not None and conf.parent is not None: self.parent = conf.parent
        subtree_conf = GNReadConf(conf)
        self.preset_conf(subtree_conf)
        with AttributeReader(element) as a: self.attributes_from_xml(a, subtree_conf)
        with OrderedTagReader(element) as r: self.children_from_xml(r, subtree_conf)

    def attributes_to_xml(self, element, conf):
        pass

    def get_child_xml_element(self, child, conf):
        return child.get_xml_element(conf)

    def get_xml_element(self, conf):
        subtree_conf = GNReadConf(conf)
        self.preset_conf(subtree_conf)
        res = etree.Element(self.tag_name(full_name = conf.parent is None or conf.parent.children_dim is None))
        self.attributes_to_xml(res, subtree_conf)
        for c in self.children:
            res.append(self.get_child_xml_element(c, subtree_conf))
        return res

    #def append(self, child):
    #    self.children.append(child)
    #    child.parent = self

    def tag_name(self, full_name = True):
        raise NotImplementedError('tag_name')

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self._parent == parent: return
        if self._parent is not None:
            self._parent.children.remove(self)
            self.in_parent = None
        self._parent = parent
        if self._parent is not None:
            self._parent.children.append(self)
