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
from numbers import Number

from . import construct_geometry_object

from .node import GNode
from .. import Info
from .object import GNObject
from ...utils.xml import attr_to_xml, xml_to_attr


class GNAgain(GNode):

    def __init__(self, parent=None, ref=None):
        super().__init__(parent)
        self.ref = ref

    def _attributes_from_xml(self, attribute_reader, reader):
        super()._attributes_from_xml(attribute_reader, reader)
        self.ref = attribute_reader.get('ref')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'ref')

    def tag_name(self, full_name=True):
        return "again"

    def stub(self):
        return ''

    def major_properties(self):
        res = super().major_properties()
        res.append(('referenced object', self.ref))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.again_copy import GNAgainController
        return GNAgainController(document, model, self)

    def real_to_model_index(self, path_iterator):
        raise IndexError()

    def create_info(self, res, names):
        super().create_info(res, names)
        if not self.ref: self._require(res, 'ref', 'referenced object ("ref")')

    #def model_to_real_index(self, index):  #TODO ??

    @staticmethod
    def from_xml(element, conf):
        result = GNAgain()
        result.load_xml_element(element, conf)
        return result

    @property
    def name(self):
        return self.ref

# ---------- copy tag and its children: --------------

class GNCopyChild(GNode):

    def __init__(self, parent=None, object=None):
        super().__init__(parent)
        self.object = object

    def _attributes_from_xml(self, attribute_reader, conf):
        self.object = attribute_reader.get('object')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'object')

    def major_properties(self):
        res = super().major_properties()
        res.append(('object', self.object))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.again_copy import GNCopyChildController
        return GNCopyChildController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        if not self.object: self._require(res, 'object')


class GNCDelete(GNCopyChild):

    def tag_name(self, full_name=True):
        return "delete"

    def get_controller(self, document, model):
        from ...controller.geometry.again_copy import GNCDeleteController
        return GNCDeleteController(document, model, self)

    def get_model_path(self, stop=None):
        path = super().get_model_path(stop)
        return path[:-1]


class GNCReplace(GNCopyChild):

    def __init__(self, parent=None, object=None, replacer=None):
        super().__init__(parent, object)
        self.replacer = replacer    # within PLaSK

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.replacer = attribute_reader.get('with')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        if self.replacer is not None: element.attrib['with'] = self.replacer

    def _children_from_xml(self, ordered_reader, conf):
        if self.replacer is None:
            dim = None
            if self.object is not None:
                obj_node = conf.find_object_by_name(self.object)
                if obj_node is not None: dim = obj_node.dim
            construct_geometry_object(ordered_reader.require(), conf, dim)

    def accept_new_child(self):
        return not self.children

    def tag_name(self, full_name=True):
        return "replace"

    def create_info(self, res, names):
        super().create_info(res, names)
        if (1 if self.replacer else 0) + len(self.children) != 1:
            self._append_info(res, 'Exactly one: "with" attribute or child must be given in {}.'.format(self.tag_name()),
                              Info.ERROR, property='replacer')

    def major_properties(self):
        res = super().major_properties()
        res.append(('with', self.replacer))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.again_copy import GNCReplaceController
        return GNCReplaceController(document, model, self)

    def model_to_real_index(self, index, model):
        return ()


class GNCToBlock(GNCopyChild):

    def __init__(self, parent=None, object=None, material=None):
        super().__init__(parent, object)
        self.name = None
        self.role = None
        self.material_top = None
        self.material_bottom = None
        self.material_shape = None

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.material_bottom = self.material_top = attribute_reader.get('material')
        if self.material_bottom is None:
            self.material_bottom = attribute_reader.get('material-bottom')
            self.material_top = attribute_reader.get('material-top')
            self.material_shape = attribute_reader.get('material-shape')
        xml_to_attr(attribute_reader, self, 'name', 'role')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        if self.material_top == self.material_bottom:
            if self.material_top is not None: element.attrib['material'] = self.material_top
        else:
            if self.material_top is not None: element.attrib['material-top'] = self.material_top
            if self.material_bottom is not None: element.attrib['material-bottom'] = self.material_bottom
            if self.material_shape is not None: element.attrib['material-shape'] = self.material_shape
        attr_to_xml(self, element, 'name', 'role')

    def tag_name(self, full_name=True):
        return "toblock"

    def major_properties(self):
        res = super().major_properties()
        if self.material_top == self.material_bottom:
            res.append(('material', self.material_bottom))
        else:
            res.append('materials:')
            res.append(('top', self.material_top))
            res.append(('bottom', self.material_bottom))
            if self.material_shape is not None: res.append(('shape', self.material_shape))
            res.append(None)
        if self.name is not None:
            res.append(('name', self.name))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.again_copy import GNCToBlockController
        return GNCToBlockController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        if not self.material_bottom or not self.material_top:
            if not self.material_bottom and not self.material_top:
                self._require(res, 'material')
            else:
                what = 'material_bottom' if not self.material_bottom else 'material_top'
                self._require(res, what, what.replace('_', ' '))

    def is_solid(self):
        return self.material_bottom == self.material_top


class GNSimplifyGradients(GNode):

    def __init__(self, parent=None, lam=None, linear=None, temp=None, only_role=None):
        self.lam = lam
        self.linear = linear
        self.temp = temp
        self.only_role = only_role
        super().__init__(parent, object)

    def tag_name(self, full_name=True):
        return "simplify-gradients"

    def get_controller(self, document, model):
        from ...controller.geometry.again_copy import GNCSimplifyGradientsController
        return GNCSimplifyGradientsController(document, model, self)

    def get_model_path(self, stop=None):
        path = super().get_model_path(stop)
        return path[:-1]

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'lam', 'temp', 'linear')
        self.only_role = attribute_reader.get('only-role')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'lam', 'temp', 'linear')
        if self.only_role is not None:
            element.attrib['only-role'] = self.only_role

    def create_info(self, res, names):
        super().create_info(res, names)
        if self.lam is None:
            self._require(res, 'lam')

    def major_properties(self):
        res = super().major_properties()
        if self.lam is not None:
            res.append(('wavelength', self.lam))
        if self.only_role is not None:
            res.append(('only-role', self.only_role))
        return res

    def get_corresponding_objects(self, manager, root, model):
        obj = root.get_object_by_model_path(manager, self.get_model_path(), model)[0]
        if self.only_role is None:
            return obj.get_matching_objects(lambda i: hasattr(i, 'roles') and '__gradient' in i.roles)
        else:
            return obj.get_matching_objects(lambda i: hasattr(i, 'roles') and '__gradient' in i.roles and self.only_role in i.roles)

class GNCopy(GNObject):

    CHANGERS = {
        'delete': GNCDelete,
        'replace': GNCReplace,
        'toblock': GNCToBlock,
        'simplify-gradients': GNSimplifyGradients,
    }

    def __init__(self, parent=None, name=None, source=None):
        super().__init__(parent, name)
        self.source = source

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.source = attribute_reader.get('from')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        if self.source is not None: element.attrib['from'] = self.source

    def _children_from_xml(self, ordered_reader, conf):
        for t in ordered_reader.iter():
            El = GNCopy.CHANGERS.get(t.tag)
            if El is None:
                ordered_reader.recent_was_unexpected()
            el = El(parent=self)
            el.load_xml_element(t, conf)

    def tag_name(self, full_name=True):
        return "copy"

    def stub(self):
        if self.name is not None and '{' not in self.name and self.source is not None and '{' not in self.source:
            return '    {} = {}'.format(self.name.replace('-', '_'), self.source)
        return ''

    def accept_new_child(self):
        return True

    def accept_as_child(self, node):
        return isinstance(node, GNCopyChild)

    def add_child_options(self):
        return [{key: lambda i1, i2: val() for key,val in GNCopy.CHANGERS.items()}]

    def get_controller(self, document, model):
        from ...controller.geometry.again_copy import GNCopyController
        return GNCopyController(document, model, self)

    def real_to_model_index(self, path_iterator):
        raise IndexError()

    def model_to_real_index(self, index, model):
        source = model.find_by_name(self.source)
        target = model.find_by_name(self.children[index].object)
        real_path = []
        have_delete = GNCDelete in (type(op) for op in self.children)
        if source is not None and target is not None:
            node = source
            model_path = target.get_model_path(source)
            for index in model_path:
                real_indexes = node.model_to_real_index(index, model)
                if isinstance(real_indexes, Number): real_indexes = [real_indexes]
                else: real_indexes = list(real_indexes)
                if have_delete and len(real_indexes) > 0:
                    children_names = [getattr(c, 'name', None) for c in node.children]
                    dec = 0
                    for op in self.children:
                        if isinstance(op, GNCDelete) and op.object is not None and op.object in children_names:
                            ci = children_names.index(op.object)
                            cri = node.model_to_real_index(ci, model)
                            if not isinstance(cri, Number): cri = cri[0]
                            if cri < real_indexes[0]:
                                dec += 1
                    real_indexes[0] -= dec
                real_path.extend(real_indexes)
                node = node.children[index]
        return real_path

    def major_properties(self):
        res = super().major_properties()
        res.append(('from', self.source))
        return res

    def create_info(self, res, names):
        super().create_info(res, names)
        if not self.source: self._require(res, 'source', '"from" attribute')

    @staticmethod
    def from_xml(element, conf):
        result = GNCopy()
        result.load_xml_element(element, conf)
        return result
