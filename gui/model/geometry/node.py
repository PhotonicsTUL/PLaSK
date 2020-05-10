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
from numbers import Number
from itertools import chain

from ...utils.validators import can_be_float
from ...utils import require_str_first_attr_path_component
from ...utils.xml import AttributeReader, OrderedTagReader
from .reader import GNReadConf, axes_dim
from .types import gname
from .. import Info
from ...utils.compat import next


class GNode:
    """Base class for all geometry nodes (objects and other XML nodes like again, copy, etc.)."""

    def __init__(self, parent=None, dim=None, children_dim=None, parent_index=None):
        """
            :param GNode parent: parent node of self (self will be added to parent's children)
            :param int dim: number of dimension of self or None if it is unknown or undefined
                   (like in case of again or copy)
            :param int children_dim: required number of dimension of self's children or None
                   if no or any children are allowed
        """
        super(GNode, self).__init__()
        self.dim = dim
        self.children_dim = children_dim
        self.children = []
        self.in_parent_aligners = None  # aligners inside the parent (container)
        self.in_parent_attrs = {}       # other attributes inside the parent (container)
        self.path = None                # path inside the parent (container)
        self._parent = None             # needed for set_parent working fine
        self.set_parent(parent, parent_index)
        self.comments = []              # comments before element in XML
        self.endcomments = []           # comments at the end of the element (in XML)
        self.itemcomments = []          # comments inside <item> this node is in (in XML)
        self.itemendcomments = []       # comments at the end of <item> this node is in (in XML)

    def _attributes_from_xml(self, attribute_reader, conf):
        """
        Read attributes of self from XML.
        This method is used by load_xml_element.
        :param GNReadConf reader: reader configuration
        :param AttributeReader attribute_reader: source of attributes
        """
        pass

    def _children_from_xml(self, ordered_reader, conf):
        """
        Read children of self from XML.
        This method is used by load_xml_element.
        :param OrderedTagReader ordered_reader: source of children XML nodes
        :param GNReadConf conf: reader configuration
        """
        pass

    def preset_conf(self, conf):
        conf.parent = self

    def load_xml_element(self, element, conf=None):
        """
        Read content of self (and whole subtree) from XML.
        Use _attributes_from_xml and _children_from_xml.
        :param etree.Element element: source XML node
        :param GNReadConf conf: reader configuration
        """
        if conf is not None and conf.parent is not None:
            self.set_parent(conf.parent, -1)
        if element is None: return
        subtree_conf = GNReadConf(conf)
        self.preset_conf(subtree_conf)
        with AttributeReader(element) as a:
            self._attributes_from_xml(a, subtree_conf)
        with OrderedTagReader(element) as r:
            self._children_from_xml(r, subtree_conf)
            self.endcomments = r.get_comments()
        self.comments = element.comments
        subtree_conf.after_read(self)

    def _attributes_to_xml(self, element, conf):
        """
        Safe XML attributes of self to element.
        This method is used by make_xml_element.
        :param etree.Element element: XML node
        :param GNReadConf conf: reader configuration
        """
        pass

    def get_child_xml_element(self, child, conf):
        """
        Get XML representation of child with position in self.
        :param GNode child: self's children
        :param GNReadConf conf: reader configuration
        :return etree.Element: XML representation of child (with whole subtree and position in self)
        """
        return child.make_xml_element(conf)

    def make_xml_element(self, conf):
        """
        Get XML representation of self.
        Use _attributes_to_xml and get_child_xml_element.
        :param GNReadConf conf: reader configuration
        :return etree.Element: XML representation of self (with children)
        """
        subtree_conf = GNReadConf(conf)
        self.preset_conf(subtree_conf)
        res = etree.Element(self.tag_name(full_name=conf.parent is None or conf.parent.children_dim is None))
        self._attributes_to_xml(res, subtree_conf)
        for child in self.children:
            for c in child.comments:
                res.append(etree.Comment(c))
            res.append(self.get_child_xml_element(child, subtree_conf))
        for c in self.endcomments:
            res.append(etree.Comment(c))
        return res

    #def append(self, child):
    #    self.children.append(child)
    #    child.parent = self

    def tag_name(self, full_name=True):
        """
        Get name of XML tag which represent self.
        :param bool full_name: only if True the full name (with 2D/3D suffix) will be returned
        :return: name of XML tag which represent self
        """
        raise NotImplementedError('tag_name')

    def display_name(self, full_name=True):
        return gname(self.tag_name(full_name=full_name))

    def accept_new_child(self):
        """
        Check if new child can be append to self.
        :return bool: True only if self can have more children.
        """
        return False

    def add_child_options(self):
        """
        Get types of children that can be added to self.
        :return:
        """
        from .types import geometry_types_2d_core_leafs, geometry_types_2d_core_containers,\
            geometry_types_2d_core_transforms, geometry_types_3d_core_leafs, geometry_types_3d_core_containers, \
            geometry_types_3d_core_transforms, geometry_types_other
        result = []
        if self.children_dim is None or self.children_dim == 2:
            result.extend((geometry_types_2d_core_leafs,
                           geometry_types_2d_core_containers,
                           geometry_types_2d_core_transforms))
        if self.children_dim is None or self.children_dim == 3:
            result.extend((geometry_types_3d_core_leafs,
                           geometry_types_3d_core_containers,
                           geometry_types_3d_core_transforms))
        result.append(geometry_types_other)
        return result

    def add_parent_options(self, current_parent=None):
        """
        Get types of children that can be added to self.
        :return:
        """
        from .types import geometry_types_2d_core_containers, geometry_types_2d_core_transforms,\
            geometry_types_3d_core_containers, geometry_types_3d_core_transforms_3d, \
            geometry_types_3d_core_extrusion, geometry_types_3d_core_revolution
        from .geometry import GNCartesian, GNCylindrical
        result = []
        if self.dim == 2 or (self.dim is None and self.parent is not None and self.parent.children_dim == 2):
            result.extend((geometry_types_2d_core_containers,
                           geometry_types_2d_core_transforms))
            if isinstance(current_parent, GNCartesian):
                result.append(geometry_types_3d_core_extrusion)
            elif isinstance(current_parent, GNCylindrical):
                result.append(geometry_types_3d_core_revolution)
        if self.dim == 3 or (self.dim is None and self.parent is not None and self.parent.children_dim == 3):
            result.extend((geometry_types_3d_core_containers,
                           geometry_types_3d_core_transforms_3d))
        return result

    def accept_as_child(self, node):
        return False

    @property
    def parent(self):
        """
        Get parent of self.
        :return GNode: parent of self
        """
        return self._parent

    @property
    def index_in_parent(self):
        return self._parent.children.index(self)

    def set_parent(self, parent, index=None, remove_from_old_parent=True, check_parent_params=True):
        """
        Move self to new parent.
        :param GNode parent: new parent of self
        :param int index: required index on new parent list (None to default)
        :param remove_from_old_parent: if True self will be removed from its current parent children list
        :param check_parent_params: if False, the `in_parent_aligners` and `path` params are kept unconditionally
        """
        if self._parent == parent:
            if index is None or parent is None: return
            # parent is not None here
            if index < 0: index = len(parent.children) + 1 - index
            if remove_from_old_parent: # move inside the current parent:
                old_index = self._parent.children.index(self)
                parent.children.insert(index, self)
                if old_index >= index: old_index += 1
                del parent.children[old_index]
            else:
                parent.children.insert(index, self)
        else:
            if self._parent is not None and remove_from_old_parent:
                self._parent.children.remove(self)
                if check_parent_params:
                    if parent is not None:
                        if type(parent) != type(self._parent):
                            self.in_parent_aligners = None
                        from .container import GNContainerBase
                        if not isinstance(parent, GNContainerBase):
                            self.path = None
                    else:
                        self.in_parent_aligners = None
                        self.path = None
            self._parent = parent
            if self._parent is not None:
                if index is None:
                    index = self._parent.new_child_pos()
                if index < 0:
                    index = len(self._parent.children) + 1 - index
                self._parent.children.insert(index, self)

    @property
    def root(self):
        return self if self._parent is None or isinstance(self._parent, GNFakeRoot) else self._parent.root

    def stub(self):
        """
        Get python stub of self (used by code completion in script section).
        :return str: python stub of self
        """
        return ''

    def get_controller(self, document, model):
        """
        Get controller which allow to change settings of self.
        :param document: document
        :param model: geometry model
        :return GNodeController: controller which allow to change settings of self
        """
        from ...controller.geometry.node import GNodeController
        return GNodeController(document, model, self)

    def get_controller_for_child_inparent(self, document, model, child):
        """
        Get controller which allow to change position self's child.
        :param document: document
        :param model: geometry model
        :param child: self's child
        :return: controller which allow to change position self's child
        """
        return None

    def get_controller_for_inparent(self, document, model):
        """
        Get controller which allows to change position of self in its parent.
        :param document: document
        :param model: geometry model
        :return: controller which allow to change position of self in its parent
        """
        if self._parent is None: return None
        return self._parent.get_controller_for_child_inparent(document, model, self)

    def major_properties(self):
        """
        Get major properties of geometry node represented by self.
        :return list: list of properties (name, value tuples). Can also include strings (to begin groups) or None-s (to end groups).
        """
        return []

    def minor_properties(self):
        """
        Get minor properties of geometry node represented by self.
        :return list: list of properties (name, value tuples). Can also include strings (to begin groups) or None-s (to end groups).
        """
        return []

    def child_properties(self, child):
        """
        Get properties of child position in self. This is typically used by containers.
        :param child: the child
        :return list: list of properties (name, value tuples). Can also include strings (to begin groups) or None-s (to end groups).
        """
        return []

    def in_parent_properties(self):
        """
        Get properties of geometry node represented by self, which are connected with its position in self.parent container.
        Call child_properties of the self.parent to do the job.
        :return list: list of properties (name, value tuples). Can also include strings (to begin groups) or None-s (to end groups).
        """
        if self._parent is None: return []
        return self._parent.child_properties(self)

    def get_axes_conf(self):
        """:return: 3D axes configuration for this node (3-elements list with name of axes)."""
        return ['long', 'tran', 'vert'] if self._parent is None else self._parent.get_axes_conf()

    def get_axes_conf_dim(self, dim=None):
        """
            :param int dim: required result dimension, self.dim by default
            :return: 2D or 3D axes configuration for this node (2 or 3 elements list with name of axes).
        """
        return axes_dim(self.get_axes_conf(), self.dim if dim is None else dim)

    def traverse(self):
        """
        Generator which visits all nodes in sub-tree fast but in undefined order.
        """
        l = [self]
        while l:
            e = l[-1]
            yield e
            l[-1:] = e.children

    def traverse_dfs(self):
        """
        Generator which visit all nodes in sub-tree in depth-first, pre-order,
        visiting children of each node from first to last.
        """
        l = [self]
        while l:
            e = l[-1]
            yield e
            l[-1:] = reversed(e.children)

    def get_names_before(self, result_set, end_node):
        """
        Search nodes in depth-first, left-to-right, pre-order and append all its names to result_set.
        Stop searching when end_node is found.
        :param set result_set: set where names are appended
        :param end_node: node which terminates searching
        :return: True only when all nodes in sub-tree has been visited and end_node has not been found
        """
        if self == end_node: return False
        name = getattr(self, 'name', None)
        if name is not None: result_set.add(name)
        for c in self.children:
            if not c.get_names_before(result_set, end_node): return False
        return True

    def get_names(self, filter=None):
        """
        Calculate all names of nodes in subtree with self in root.
        :param filter: names filter
        :return set: calculated set of names
        """
        if filter is not None:
            return set(n for n in (getattr(nd, 'name', None) for nd in self.traverse() if filter(nd)) if n is not None)
        else:
            return set(n for n in (getattr(nd, 'name', None) for nd in self.traverse()) if n is not None)

    def find(self, predicate):
        """
        Find the node which fulfil given predicate in subtree with self in root.
        :param predicate: callable, predicate(node), where node is GNode, should return bool, True to accept
        :return GNode: node with name attribute equals to name argument or None if not found
        """
        for node in self.traverse():
            if predicate(node): return node
        return None

    def find_by_name(self, name):
        """
        Find the node with given name in subtree with self in root.
        :param str name: name to find
        :return GNode: node with name attribute equals to name argument or None if not found
        """
        return self.find(lambda node: getattr(node, 'name', None) == name)

    def get_paths(self, filter=None):
        """
        Calculate all path's names of nodes in subtree with self in root.
        :return set: calculated set of path's names
        """
        if filter is not None:
            return set(chain(*(
                (p.strip() for p in ps.split(',')) for ps in
                (getattr(nd, 'path', None) for nd in self.traverse() if filter(nd))
                if ps is not None)))
        else:
            return set(chain(*(
                (p.strip() for p in ps.split(',')) for ps in
                (getattr(nd, 'path', None) for nd in self.traverse())
                if ps is not None)))

    def new_child_pos(self):
        """Return position of a new child"""
        return len(self.children)

    @property
    def path_to_root(self):
        p = self
        while p is not None:
            yield p
            p = p.parent

    # def get_child_by_real_index(self, index):
    #     """
    #     Get child of self indexed by GeometryObject's index.
    #     :param int index: index of child of the GeometryObject represented by self
    #     :return GNode: child of self which represents corresponding child of the GeometryObject
    #     """
    #     return self.children[index]

    def real_to_model_index(self, path_iterator):
        """
        Calculate model index which corresponds to beginning of the real path.
        :param path_iterator: iterator over path
        :return int: index
        """
        return next(path_iterator)

    def model_to_real_index(self, index, model):
        """
        Calculate real index (path fragment) of child with given model index
        :param int index: model index
        :param model: geometry model
        :return: path fragment, sequence of indexes
        """
        return index,

    def get_node_by_real_path(self, real_path):
        """
        Go downside in the model tree following real plask path as long as it is possible.
        :param real_path: collection of indexes in GeometryObjects' tree
        :return GNode: achieved node
        """
        node = self
        real_path_iterator = iter(real_path)
        while True:
            try:
                node = node.children[node.real_to_model_index(real_path_iterator)]
            except (StopIteration, IndexError):
                break
        return node

    def get_object_by_model_path(self, object, model_path, model):
        """
        :param plask.GeometryObject object: object which is represented by self
               or plask.Manager if model_path is absolute
        :param collection.Iterable model_path: collection of indexes in GNode's tree
        :param model: geometry model
        :return plask.GeometryObject, GNode: object and node which represents this object
        """
        node = self
        for index in model_path:
            real_indexes = node.model_to_real_index(index, model)
            if isinstance(real_indexes, Number): real_indexes = (real_indexes,)
            try:
                for real_index in real_indexes:
                    try:
                        object = object[real_index]
                    except TypeError:  # geometry manager has no child:
                        object = object._roots[real_index]
                node = node.children[index]
            except (IndexError, AttributeError):
                return None, None
        return object, node

    def get_model_path(self, stop=None):
        """
        :return: path in model from fake_root to self
        """
        if self.parent is None or self == stop:
            return []
        result = self.parent.get_model_path(stop)
        result.append(self.index_in_parent)
        return result

    def _append_info(self, res, text, level=None, nodes=None, **kwargs):
        if nodes is None: nodes = (self,)
        res.append(Info(text, level, nodes=nodes, **kwargs))

    def _append_error(self, res, text, nodes=None, **kwargs):
        self._append_info(res, text, Info.ERROR, nodes, **kwargs)

    def _require(self, res, property, display_name=None, indexes=None, type=None):
        if display_name is None: display_name = '"{}"'.format(require_str_first_attr_path_component(property))
        if type is not None: display_name = 'valid {} value for {}'.format(type, display_name)
        self._append_error(res, 'Specifying {} is required in <{}>'.format(display_name, self.tag_name(False)),
                           property=property, indexes=indexes)

    def _wrong_type(self, res, type, property, display_name=None, indexes=None):
        if display_name is None: display_name = '"{}"'.format(require_str_first_attr_path_component(property))
        self._append_error(res, '{} has not valid {} value in <{}>'.format(display_name, type, self.tag_name(False)),
                           property=property, indexes=indexes)

    def _require_child(self, res):
        if not self.children:
            self._append_error(res, '<{}> requires an item'.format(self.tag_name(False)))

    def create_info(self, res, names):
        """
        :param List(Info) res: place to append info objects
        :param OrderedDict names: names of objects which are before this in tree
        """
        if not can_be_float(self.in_parent_attrs.get('zero')): self._wrong_type(res, 'float', 'zero')


class GNFakeRoot(GNode):

    def __init__(self, geometry_model):
        super(GNFakeRoot, self).__init__()
        self.model = geometry_model

    def accept_new_child(self):
        return True

    def accept_as_child(self, node):
        from .geometry import GNGeometryBase
        return isinstance(node, GNGeometryBase)

    def tag_name(self, full_name=True):
        return "geometry"

    def get_corresponding_object(self, node, manager):
        """
        Get object that corresponds to node in real objects tree.
        :param GNode node:
        :param plask.Manager manager: manager which describes real geometry objects tree
        :return plask.GeometryObject: object that corresponds to self in real objects tree
        """
        try:
            return self.get_object_by_model_path(manager, node.get_model_path(), self.model)[0]
        except (IndexError, ValueError, TypeError):
            return None