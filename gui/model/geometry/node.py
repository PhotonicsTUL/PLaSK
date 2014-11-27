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

from .reader import GNReadConf, axes_dim
from ...utils.xml import AttributeReader, OrderedTagReader


class GNode(object):

    """
        :param parent: parent node of self (self will be added to parent's children)
        :param dim: number of dimension of self or None if it is unknown or not defined (like in case of again or copy)
        :param children_dim: required number of dimension of self's children or None if no or any children are allowed
    """
    def __init__(self, parent=None, dim=None, children_dim=None):
        super(GNode, self).__init__()
        self.dim = dim
        self.children_dim = children_dim
        self.children = []
        self.in_parent = None   #configuration inside parent (container)
        self.path = None    #path inside parent (container)
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

    def set_xml_element(self, element, conf=None):
        if conf is not None and conf.parent is not None: self.parent = conf.parent
        if element is None: return
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
        res = etree.Element(self.tag_name(full_name=conf.parent is None or conf.parent.children_dim is None))
        self.attributes_to_xml(res, subtree_conf)
        for c in self.children:
            res.append(self.get_child_xml_element(c, subtree_conf))
        return res

    #def append(self, child):
    #    self.children.append(child)
    #    child.parent = self

    def tag_name(self, full_name=True):
        raise NotImplementedError('tag_name')

    def accept_new_child(self):
        return False

    def add_child_options(self):
        from .types import geometry_types_2d_core_leafs, geometry_types_2d_core_containers, geometry_types_2d_core_transforms,\
                           geometry_types_3d_core_leafs, geometry_types_3d_core_containers, geometry_types_3d_core_transforms,\
                           geometry_types_other
        result = []
        if self.children_dim is None or self.children_dim == 2:
            result.extend((geometry_types_2d_core_containers, geometry_types_2d_core_leafs, geometry_types_2d_core_transforms))
        if self.children_dim is None or self.children_dim == 3:
            result.extend((geometry_types_3d_core_containers, geometry_types_3d_core_leafs, geometry_types_3d_core_transforms))
        result.append(geometry_types_other)
        return result

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self._parent == parent: return
        if self._parent is not None:
            self._parent.children.remove(self)
            self.in_parent = None
            self.path = None
        self._parent = parent
        if self._parent is not None:
            self._parent.children.append(self)

    def stub(self):
        return ''

    def get_controller(self, document, model):
        from ...controller.geometry.node import GNodeController
        return GNodeController(document, model, self)

    def get_controller_for_child_inparent(self, document, model, child):
        return None

    def get_controller_for_inparent(self, document, model):
        if self._parent is None: return None
        return self._parent.get_controller_for_child_inparent(document, model, self)

    def major_properties(self):
        '''
        Get major properties of geometry node represented by self.
        :return list: list of properties (name, value tuples). Can also include strings (to begin groups) or None-s (to end groups).
        '''
        return []

    def minor_properties(self):
        '''
        Get minor properties of geometry node represented by self.
        :return list: list of properties (name, value tuples). Can also include strings (to begin groups) or None-s (to end groups).
        '''
        return []

    def child_properties(self, child):
        '''
        Get properties of child position in self. This is typically used by containers.
        :param child: the child
        :return list: list of properties (name, value tuples). Can also include strings (to begin groups) or None-s (to end groups).
        '''
        return []

    def in_parent_properties(self):
        '''
        Get properties of geometry node represented by self, which are connected with its position in self.parent container.
        Call child_properties of the self.parent to do the job.
        :return list: list of properties (name, value tuples). Can also include strings (to begin groups) or None-s (to end groups).
        '''
        if self._parent is None: return []
        return self._parent.child_properties(self)

    def get_axes_conf(self):
        ''':return: Axes configuration for this node (3D list with name of axes).'''
        return ['z', 'x', 'y'] if self._parent is None else self._parent.get_axes_conf()

    def get_axes_conf_dim(self, dim=None):
        return axes_dim(self.get_axes_conf(), self.dim if dim is None else dim)

    def traverse_dfs(self):
        '''
        Generator which visit all nodes in sub-tree in depth-first order
        :return: next calls return next nodes in depth-first order
        '''
        yield self
        for c in self.children:
            for t in c.traverse_dfs():
                yield t

    def traverse_dfs_fun(self, f):
        '''
        Visit all nodes in sub-tree in depth-first order and call f for each.
        :param f: function to call for each node, should return True to continue searching and False to stop
        :return: True if f returns True for all nodes in subtree, False in other cases
        '''
        if f(self):
            for c in self.children:
                if not c.traverse_dfs_fun(f):
                    return False
            return True
        else:
            return False

    def names_before(self, result_set, end_node):
        if self == end_node: return False
        name = getattr(self, 'name', None)
        if name is not None: result_set.add(name)
        for c in self.children:
            if not c.names_before(result_set, end_node): return False
        return True

    def names(self):
        return set(n for n in (getattr(nd, 'name', None) for nd in self.traverse_dfs()) if n is not None)

    def paths(self):
        return set(n for n in (getattr(nd, 'path', None) for nd in self.traverse_dfs()) if n is not None)