# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import weakref
from lxml import etree

from .reader import GNAligner
from .object import GNObject
from .node import GNode
from .constructor import construct_geometry_object
from ...utils import get_manager
from ...utils.validators import can_be_float, can_be_int, can_be_bool, can_be_one_of
from ...utils.xml import AttributeReader, OrderedTagReader, xml_to_attr, attr_to_xml
from ...utils.config import CONFIG
from ...utils.str import empty_to_none

class GNZero(GNode):

    def __init__(self, parent=None, dim=None, parent_index=None):
        super().__init__(parent=parent, dim=dim, parent_index=parent_index)

    def tag_name(self, full_name=True):
        return 'zero'

    @staticmethod
    def from_xml(element, conf):
        result = GNZero()
        result.load_xml_element(element, conf)
        return result

    def add_parent_options(self, current_parent=None):
        return []


class GNGap(GNode):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=2)
        self.size = None
        self.size_is_total = False

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.size = attribute_reader.get('size')
        if self.size is not None:
            self.size_is_total = False
        else:
            self.size = attribute_reader.get('total')
            if self.size is not None: self.size_is_total = True

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        if self.size is not None:
            element.attrib['total' if self.size_is_total else 'size'] = self.size

    def tag_name(self, full_name=True):
        return 'gap'

    def get_controller(self, document, model):
        from ...controller.geometry.container import GNGapController
        return GNGapController(document, model, self)

    def major_properties(self):
        res = super().major_properties()
        if self.size is not None:
            res.append(('total container size' if self.size_is_total else 'gap size', self.size))
        return res

    def create_info(self, res, names):
        super().create_info(res, names)
        if not can_be_float(self.size, required=True, float_validator=lambda f: f >= 0):
            self._require(res, 'size', 'gap or total size', type='non-negative float')

    @staticmethod
    def from_xml(element, conf):
        result = GNGap()
        result.load_xml_element(element, conf)
        return result

    def add_parent_options(self, current_parent=None):
        return []


class GNContainerBase(GNObject):

    def accept_new_child(self):
        return True

    def accept_as_child(self, node):
        from .geometry import GNGeometryBase
        if isinstance(node, GNGeometryBase): return False
        from .again_copy import GNCopy, GNAgain
        from .python_object import GNPython
        return (isinstance(node, GNObject) and node.dim == self.children_dim) or \
                isinstance(node, GNCopy) or isinstance(node, GNAgain) or isinstance(node, GNPython)

    def aligners_dir(self):
        return None

    def get_controller_for_child_inparent(self, document, model, child):
        from ...controller.geometry.container import GNContainerChildController
        return GNContainerChildController(document, model, self, child)

    def child_properties(self, child):
        res = super().child_properties(child)
        res.append(('path', child.path))
        return res

    def item_attributes_from_xml(self, child, item_attr_reader, conf):
        """
        Read item tag attributes from XML.
        Default implementation supports path attribute.
        :param child: child node
        :param item_attr_reader: element-tree attributes reader
        :param GNReadConf conf: read configuration (axes, etc.)
        """
        child.path = item_attr_reader.get('path')

    def child_from_xml(self, child_element, conf):
        """
        Read single child from XML.
        Default implementation supports item tag (call item_attributes_from_xml to read it attributes)
        and call construct_geometry_object(child_element, conf) for all other tags.
        Subclasses should support custom tags before call this implementation.
        :param child_element: element-tree element with child configuration
        :param GNReadConf conf: read configuration (axes, etc.)
        :return GNode: child node
        """
        if child_element.tag == 'item':
            with OrderedTagReader(child_element) as item_child_reader:
                child = construct_geometry_object(item_child_reader.require(), conf)
                child.comments, child.itemcomments = child_element.comments, child.comments
                child.itemendcomments = item_child_reader.get_comments()
            with AttributeReader(child_element) as item_attr_reader:
                self.item_attributes_from_xml(child, item_attr_reader, conf)
        else:
            construct_geometry_object(child_element, conf)

    def _children_from_xml(self, ordered_reader, conf):
        """Call child_from_xml for each child. Subclasses shouldn't overwrite this but child_from_xml instead."""
        for child_element in ordered_reader.iter():
            self.child_from_xml(child_element, conf)

    def get_item_xml_element(self, child, conf):
        """
        Construct empty (but with attributes) item XML tag which describe child position in self.
        Default implementation support path attribute.
        :param GNode child: children which parent-related attributes should be written to returned element
        :param GNReadConf conf: read configuration (axes, etc.)
        :return etree.Element: item tag (without children)
        """
        res = etree.Element('item')
        if child.path is not None: res.attrib['path'] = child.path
        return res

    def get_child_xml_element(self, child, conf):
        """
        Subclasses should not overwrite this but some of get_item_xml_element, item_xml_element_is_required or attributes_to_xml instead
        (all method mentioned are used by this).
        :param GNode child: child to get element for
        :param GNReadConf conf: read configuration (axes, etc.)
        :return etree.Element: XML element which represent child (possibly item tag)
        """
        child_element = super().get_child_xml_element(child, conf)
        res = self.get_item_xml_element(child, conf)
        if res.attrib or child.itemcomments or child.itemendcomments:
            for c in child.itemcomments:
                res.append(etree.Comment(c))
            res.append(child_element)
            for c in child.itemendcomments:
                res.append(etree.Comment(c))
            return res
        else:
            return child_element

    def model_to_real_index(self, index, model):
        return index, 0

    def real_to_model_index(self, path_iterator):
        index = next(path_iterator)
        next(path_iterator)    # skip 0
        return index


class GNStackBase(GNContainerBase):

    def item_attributes_from_xml(self, child, item_attr_reader, conf):
        super().item_attributes_from_xml(child, item_attr_reader, conf)
        child.in_parent_attrs['zero'] = item_attr_reader.get('zero')

    def get_item_xml_element(self, child, conf):
        res = super().get_item_xml_element(child, conf)
        zero = child.in_parent_attrs.get('zero')
        if zero is not None: res.attrib['zero'] = zero
        return res

    def get_controller_for_child_inparent(self, document, model, child):
        from ...controller.geometry.container import GNStackChildController
        return GNStackChildController(document, model, self, child)


class GNStack(GNStackBase):
    """2D/3D (multi-)stack"""

    def __init__(self, parent=None, dim=None):
        super().__init__(parent=parent, dim=dim, children_dim=dim)
        self.repeat = None
        self.shift = None
        self.aligners = [GNAligner(None, None) for _ in range(0, dim-1)]

    def accept_as_child(self, node):
        return super().accept_as_child(node) or isinstance(node, GNZero)

    def aligners_dir(self):
        return (0,) if self.children_dim == 2 else (0, 1)

    def aligners_dict(self, aligners=None):
        if not aligners: aligners = self.aligners
        return dict(zip(self.aligners_dir(), aligners))

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.repeat = attribute_reader.get('repeat')
        self.shift = attribute_reader.get('shift')
        self.aligners = conf.read_aligners(attribute_reader, self.children_dim, *self.aligners_dir())

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'repeat', 'shift')
        conf.write_aligners(element, self.children_dim, self.aligners_dict())

    def item_attributes_from_xml(self, child, item_attr_reader, conf):
        super().item_attributes_from_xml(child, item_attr_reader, conf)
        child.in_parent_aligners = conf.read_aligners(item_attr_reader, self.children_dim, *self.aligners_dir())

    def child_from_xml(self, child_element, conf):
        if child_element.tag == 'zero':
            GNZero(self, self.children_dim, parent_index=-1)
        else:
            super().child_from_xml(child_element, conf)

    def get_item_xml_element(self, child, conf):
        res = super().get_item_xml_element(child, conf)
        if child.in_parent_aligners is not None:
            conf.write_aligners(res, self.children_dim, self.aligners_dict(child.in_parent_aligners))
        return res

    def tag_name(self, full_name=True):
        return "stack{}d".format(self.dim) if full_name else "stack"

    def python_type(self):
        if self.repeat is None or self.repeat.strip() == '1':
            return 'geometry.SingleStack{}D'.format(self.dim)
        else:
            return 'geometry.MultiStack{}D'.format(self.dim)

    def create_info(self, res, names):
        super().create_info(res, names)
        zeros = tuple(z for z in self.children if isinstance(z, GNZero))
        zeros_num = len(zeros) + (1 if self.shift else 0)
        if zeros_num > 1: self._append_error(res,
                'Position of the stack bottom edge is defined {} times in {}.'.format(zeros_num, self.tag_name(False)),
                nodes=(self,)+zeros)
        if not can_be_float(self.shift): self._wrong_type(res, 'float', 'shift')
        if not can_be_int(self.repeat, int_validator=lambda i: i >= 0): self._wrong_type(res, 'non-negative integer', 'repeat')

    def add_child_options(self):
        res = super().add_child_options()
        res.insert(0, {'zero': GNZero.from_xml})
        return res

    def _aligner_to_property(self, aligner_dir, aligner):
        return (aligner.position_str(self.children_dim, self.get_axes_conf_dim(self.children_dim), aligner_dir), aligner.value)

    def _aligners_to_properties(self, aligners):
        res = []
        if aligners is None: return res
        for aligner_dir, aligner in zip(self.aligners_dir(), aligners):
            if aligner.position is not None:
                res.append(self._aligner_to_property(aligner_dir, aligner))
        return res

    def major_properties(self):
        res = super().major_properties()
        res += self._aligners_to_properties(self.aligners)
        res.append(('repeat', self.repeat))
        res.append(('shift', self.shift))
        return res

    def child_properties(self, child):
        res = super().child_properties(child)
        return res + self._aligners_to_properties(child.in_parent_aligners)

    def get_controller(self, document, model):
        from ...controller.geometry.container import GNStackController
        return GNStackController(document, model, self)

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNStack(dim=2)
        result.load_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNStack(dim=3)
        result.load_xml_element(element, conf)
        return result

    def new_child_pos(self):
        return 0

    @property
    def real_children_count(self):
        return sum(1 for c in self.children if not isinstance(c, GNZero))

    def real_to_model_index(self, path_iterator):
        index = super().real_to_model_index(path_iterator)
        real_count = self.real_children_count
        index = real_count - 1 - index % real_count  # reduce to the first period and change to model order
        i = 0   # considering GNZero nodes:
        while i <= index:
            if isinstance(self.children[i], GNZero): index += 1
            i += 1
        return index

    def model_to_real_index(self, index, model):
        #if isinstance(self.children[index], GNZero)    #TODO throw exception
        index -= sum(1 for i in range(0, index) if isinstance(self.children[i], GNZero))
        return super().model_to_real_index(self.real_children_count - 1 - index, model)


class GNShelf(GNStackBase):
    """(multi-)shelf"""

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=2, children_dim=2)
        self.repeat = None
        self.shift = None
        self.flat = None

    def accept_as_child(self, node):
        return super().accept_as_child(node) or isinstance(node, GNZero) or isinstance(node, GNGap)

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'repeat', 'shift', 'flat')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'repeat', 'shift', 'flat')

    def child_from_xml(self, child_element, conf):
        if child_element.tag == 'zero':
            GNZero(self, self.children_dim)
        elif child_element.tag == 'gap':
            GNGap.from_xml(child_element, conf)
        else:
            super().child_from_xml(child_element, conf)

    def tag_name(self, full_name=True):
        return "shelf{}d".format(self.dim) if full_name else "shelf"

    def python_type(self):
        return 'geometry.Shelf2D{}'

    def add_child_options(self):
        res = super().add_child_options()
        res.insert(0, {'gap': GNGap.from_xml, 'zero': GNZero.from_xml})
        return res

    def major_properties(self):
        res = super().major_properties()
        res.append(('repeat', self.repeat))
        res.append(('shift', self.shift))
        return res

    def minor_properties(self):
        res = super().minor_properties()
        res.append(('flat', self.flat))
        return res

    def create_info(self, res, names):
        super().create_info(res, names)
        zeros = tuple(z for z in self.children if isinstance(z, GNZero))
        zeros_num = len(zeros) + (1 if self.shift else 0)
        if zeros_num > 1: self._append_error(res,
                'Position of the shelf left edge is defined {} times.'.format(zeros_num),
                nodes=(self,)+zeros)
        gap_with_total = tuple(g for g in self.children if isinstance(g, GNGap) and g.size_is_total)
        if len(gap_with_total) > 1:
            self._append_error(res,
                '{} gaps have been given total shelf size.'.format(len(gap_with_total)),
                nodes=(self,)+gap_with_total)
        if not can_be_float(self.shift): self._wrong_type(res, 'float', 'shift')
        if not can_be_int(self.repeat, int_validator=lambda i: i >= 0): self._wrong_type(res, 'non-negative integer', 'repeat')
        if not can_be_bool(self.flat): self._wrong_type(res, 'boolean', 'flat')

    def get_controller(self, document, model):
        from ...controller.geometry.container import GNShelfController
        return GNShelfController(document, model, self)

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNShelf()
        result.load_xml_element(element, conf)
        return result


class GNAlignContainer(GNContainerBase):
    """2D/3D align container"""

    def __init__(self, parent=None, dim=None):
        super().__init__(parent=parent, dim=dim, children_dim=dim)
        self.order = empty_to_none(CONFIG['geometry/default_align_ordering'])
        self.aligners = [GNAligner(None, None) for _ in range(0, self.children_dim)]
        self._document = None

    def aligners_dir(self):
        return range(0, self.children_dim)

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'order')
        self.aligners = conf.read_aligners(attribute_reader, self.children_dim)

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'order')
        conf.write_aligners(element, self.children_dim, self.aligners)

    def item_attributes_from_xml(self, child, item_attr_reader, conf):
        super().item_attributes_from_xml(child, item_attr_reader, conf)
        child.in_parent_aligners = conf.read_aligners(item_attr_reader, self.children_dim)

    def get_item_xml_element(self, child, conf):
        res = super().get_item_xml_element(child, conf)
        if child.in_parent_aligners is not None:
            conf.write_aligners(res, self.children_dim, child.in_parent_aligners)
        return res

    def tag_name(self, full_name=True):
        return "align{}d".format(self.dim) if full_name else "align"

    def python_type(self):
        return 'geometry.Align{}D'.format(self.dim)

    def _aligners_to_properties(self, aligners):
        res = []
        axes_conf = self.get_axes_conf_dim(self.children_dim)
        for axis_nr, aligner in enumerate(aligners):
            if aligner.position is not None:
                res.append((aligner.position_str(self.children_dim, axes_conf, axis_nr), aligner.value))
        return res

    def create_info(self, res, names):
        super().create_info(res, names)
        if not can_be_one_of(self.order, 'normal', 'reverse'):
            self._append_error(res, "order (in <align>) must be 'normal' or 'reverse'".format(self.tag_name()))
        for i, a in enumerate(self.aligners):
            if not can_be_float(a.value): self._wrong_type(res, 'float', ('positions', i, 1), 'component of default items position')

    def major_properties(self):
        return super().major_properties() + self._aligners_to_properties(self.aligners)

    def child_properties(self, child):
        if child.in_parent_aligners is None: return []
        return self._aligners_to_properties(child.in_parent_aligners)

    def get_controller(self, document, model):
        if isinstance(document, weakref.ProxyType):
            self._document = document
        else:
            self._document = weakref.proxy(document)
        from ...controller.geometry.container import GNAlignContainerController
        return GNAlignContainerController(document, model, self)

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNAlignContainer(dim=2)
        result.load_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNAlignContainer(dim=3)
        result.load_xml_element(element, conf)
        return result

    @property
    def _reverse(self):
        if self.order is None: return False
        if '{' not in self.order:
            return self.order == 'reverse'
        try:
            manager = get_manager()
            manager.load(self._document.get_contents(sections=('defines')))
            try:
                order = eval(f'f"{self.order}"', manager.defs)
            except SyntaxError:
                order = self.order.format(**manager.defs)
            return order == 'reverse'
        except Exception:
            return False

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_document'] = None
        return state

    def real_to_model_index(self, path_iterator):
        index = super().real_to_model_index(path_iterator)
        if self._reverse:
            children_count = len(self.children)
            index = (children_count - 1 - index) % children_count
        return index

    def model_to_real_index(self, index, model):
        if self._reverse:
            children_count = len(self.children)
            index = (children_count - 1 - index) % children_count
        return super().model_to_real_index(index, model)
