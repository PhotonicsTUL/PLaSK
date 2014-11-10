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

from .reader import GNAligner
from .object import GNObject
from .node import GNode
from .constructor import construct_geometry_object
from ...utils.xml import AttributeReader, OrderedTagReader, xml_to_attr, attr_to_xml


class GNZero(GNode):

    def __init__(self, parent = None, dim = None):
        super(GNZero, self).__init__(parent=parent, dim=dim)

    def tag_name(self, full_name = True):
        return 'zero'

    @classmethod
    def from_xml(cls, element, conf):
        result = GNZero()
        result.set_xml_element(element, conf)
        return result


class GNGap(GNode):

    def __init__(self, parent = None):
        super(GNGap, self).__init__(parent=parent, dim=2)
        self.size = None
        self.size_is_total = False

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNGap, self).attributes_from_xml(attribute_reader, conf)
        self.size = attribute_reader.get('size')
        if self.size is not None:
            self.size_is_total = False
        else:
            self.size = attribute_reader.get('total')
            if self.size is not None: self.size_is_total = True

    def attributes_to_xml(self, element, conf):
        super(GNGap, self).attributes_to_xml(element, conf)
        if self.size is not None:
            element.attrib['total' if self.size_is_total else 'size'] = self.size

    def tag_name(self, full_name = True):
        return 'gap'

    def get_controller(self, document, model):
        from ...controller.geometry.container import GNGapController
        return GNGapController(document, model, self)

    def major_properties(self):
        res = super(GNGap, self).major_properties()
        if self.size is not None:
            res.append(('total container size' if self.size_is_total else 'gap size', self.size))
        return res

    @classmethod
    def from_xml(cls, element, conf):
        result = GNGap()
        result.set_xml_element(element, conf)
        return result


class GNContainerBase(GNObject):

    def accept_new_child(self):
        return True


class GNStack(GNContainerBase):
    """2D/3D (multi-)stack"""

    def __init__(self, parent = None, dim = None):
        super(GNStack, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.repeat = None
        self.shift = None
        self.aligners = (GNAligner(None, None) for _ in range(0, dim-1))

    def aligners_dir(self):
        return (0,) if self.children_dim == 2 else (0, 1)

    def aligners_dict(self):
        return dict(zip(self.aligners_dir(), self.aligners))

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNStack, self).attributes_from_xml(attribute_reader, conf)
        self.repeat = attribute_reader.get('repeat')
        self.shift = attribute_reader.get('shift')
        self.aligners = conf.read_aligners(attribute_reader, self.children_dim, *self.aligners_dir())

    def attributes_to_xml(self, element, conf):
        super(GNStack, self).attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'repeat', 'shift')
        conf.write_aligners(element, self.children_dim, self.aligners_dict())

    def children_from_xml(self, ordered_reader, conf):
        for c in ordered_reader.iter():
            if c.tag == 'item':
                with OrderedTagReader(c) as item_child_reader:
                    child = construct_geometry_object(item_child_reader.require(), conf)
                with AttributeReader(c) as item_attr_reader:
                    child.in_parent, = conf.read_aligners(item_attr_reader, self.children_dim, self.aligners_dir())  #self.children_dim-2 is direction tran
            elif c.tag == 'zero':
                GNZero(self, self.children_dim)
            else:
                construct_geometry_object(c, conf)

    def get_child_xml_element(self, child, conf):
        child_element = super(GNStack, self).get_child_xml_element(child, conf)
        if child.in_parent is not None:
            res = etree.Element('item')
            res.append(child_element)
            conf.write_aligners(res, self.children_dim, self.aligners_dict())
            return res
        else:
            return child_element

    def tag_name(self, full_name = True):
        return "stack{}d".format(self.dim) if full_name else "stack"

    def python_type(self):
        if self.repeat is None or self.repeat.strip() == '1':
            return 'geometry.SingleStack{}D'.format(self.dim)
        else:
            return 'geometry.MultiStack{}D'.format(self.dim)

    def add_child_options(self):
        res = super(GNStack, self).add_child_options()
        res.insert(0, {'zero': GNZero.from_xml})
        return res

    def _aligner_to_property(self, aligner_dir, aligner):
        return (aligner.position_str(self.children_dim, self.get_axes_conf_dim(self.children_dim), aligner_dir), aligner.value)

    def major_properties(self):
        res = super(GNStack, self).major_properties()
        for aligner_dir, aligner in zip(self.aligners_dir(), self.aligners):
            if aligner.position is not None:
                res.append(self._aligner_to_property(aligner_dir, aligner))
        res.append(('repeat', self.repeat))
        res.append(('shift', self.shift))
        return res

    def child_properties(self, child_in_parent):
        if child_in_parent is None or child_in_parent.position is None: return []
        return [self._aligner_to_property(child_in_parent)]

    def get_controller(self, document, model):
        from ...controller.geometry.container import GNStackController
        return GNStackController(document, model, self)

    @classmethod
    def from_xml_2d(cls, element, conf):
        result = GNStack(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(cls, element, conf):
        result = GNStack(dim = 3)
        result.set_xml_element(element, conf)
        return result


class GNShelf(GNContainerBase):
    """(multi-)shelf"""

    def __init__(self, parent = None):
        super(GNShelf, self).__init__(parent=parent, dim=2, children_dim=2)
        self.repeat = None
        self.shift = None
        self.flat = None

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNShelf, self).attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'repeat', 'shift', 'flat')

    def attributes_to_xml(self, element, conf):
        super(GNShelf, self).attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'repeat', 'shift', 'flat')

    def children_from_xml(self, ordered_reader, conf):
        for c in ordered_reader.iter():
            if c.tag == 'zero':
                GNZero(self, self.children_dim)
            elif c.tag == 'gap':
                GNGap.from_xml(c, conf)
            else:
                construct_geometry_object(c, conf)

    def tag_name(self, full_name = True):
        return "shelf{}d".format(self.dim) if full_name else "shelf"

    def python_type(self):
        return 'geometry.Shelf2D{}'

    def add_child_options(self):
        res = super(GNShelf, self).add_child_options()
        res.insert(0, {'gap': GNGap.from_xml, 'zero': GNZero.from_xml})
        return res

    def major_properties(self):
        res = super(GNShelf, self).major_properties()
        res.append(('repeat', self.repeat))
        res.append(('shift', self.shift))
        return res

    def minor_properties(self):
        res = super(GNShelf, self).minor_properties()
        res.append(('flat', self.flat))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.container import GNShelfController
        return GNShelfController(document, model, self)

    @classmethod
    def from_xml_2d(cls, element, conf):
        result = GNShelf()
        result.set_xml_element(element, conf)
        return result


class GNAlignContainer(GNContainerBase):
    """2D/3D align container"""

    def __init__(self, parent = None, dim = None):
        super(GNAlignContainer, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.aligners = [GNAligner(None, None) for _ in range(0, self.children_dim)]

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNAlignContainer, self).attributes_from_xml(attribute_reader, conf)
        self.aligners = conf.read_aligners(attribute_reader, self.children_dim)

    def attributes_to_xml(self, element, conf):
        super(GNAlignContainer, self).attributes_to_xml(element, conf)
        conf.write_aligners(element, self.children_dim, self.aligners)

    def children_from_xml(self, ordered_reader, conf):
        for c in ordered_reader.iter():
            if c.tag == 'item':
                with OrderedTagReader(c) as item_child_reader:
                    child = construct_geometry_object(item_child_reader.require(), conf)
                with AttributeReader(c) as item_attr_reader:
                    child.in_parent = conf.read_aligners(item_attr_reader, self.children_dim)
            else:
                construct_geometry_object(c, conf)

    def get_child_xml_element(self, child, conf):
        child_element = super(GNAlignContainer, self).get_child_xml_element(child, conf)
        if child.in_parent is not None:
            res = etree.Element('item')
            res.append(child_element)
            conf.write_aligners(res, self.children_dim, child.in_parent)
            return res
        else:
            return child_element

    def tag_name(self, full_name = True):
        return "align{}d".format(self.dim) if full_name else "align"

    def python_type(self):
        return 'geometry.AlignContainer{}D'.format(self.dim)

    def _aligners_to_properties(self, aligners):
        res = []
        axes_conf = self.get_axes_conf_dim(self.children_dim)
        for axis_nr, aligner in enumerate(aligners):
            if aligner.position is not None:
                res.append((aligner.position_str(self.children_dim, axes_conf, axis_nr), aligner.value))
        return res

    def major_properties(self):
        return super(GNAlignContainer, self).major_properties() + self._aligners_to_properties(self.aligners)

    def child_properties(self, child_in_parent):
        if child_in_parent is None: return []
        return self._aligners_to_properties(child_in_parent)

    @classmethod
    def from_xml_2d(cls, element, conf):
        result = GNAlignContainer(dim = 2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(cls, element, conf):
        result = GNAlignContainer(dim = 3)
        result.set_xml_element(element, conf)
        return result