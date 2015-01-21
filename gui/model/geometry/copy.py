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
from . import construct_geometry_object
from .reader import GNReadConf

from .node import GNode
from .object import GNObject
from ...utils.xml import attr_to_xml


class GNAgain(GNode):

    def __init__(self, parent=None, ref=None):
        super(GNAgain, self).__init__(parent)
        self.ref = ref

    def _attributes_from_xml(self, attribute_reader, reader):
        super(GNAgain, self)._attributes_from_xml(attribute_reader, reader)
        self.ref = attribute_reader.get('ref')

    def _attributes_to_xml(self, element, conf):
        super(GNAgain, self)._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'ref')

    def tag_name(self, full_name=True):
        return "again"

    def stub(self):
        return ''

    def major_properties(self):
        res = super(GNAgain, self).major_properties()
        res.append(('referenced object', self.ref))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.copy import GNAgainController
        return GNAgainController(document, model, self)

    @classmethod
    def from_xml(cls, element, conf):
        result = GNAgain()
        result.set_xml_element(element, conf)
        return result


# ---------- copy tag and its children: --------------

class GNCopyChild(GNode):

    def __init__(self, parent=None, object=None):
        super(GNCopyChild, self).__init__(parent)
        self.object = object

    def _attributes_from_xml(self, attribute_reader, conf):
        self.object = attribute_reader.get('object')

    def _attributes_to_xml(self, element, conf):
        super(GNCopyChild, self)._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'object')

    def major_properties(self):
        res = super(GNCopyChild, self).major_properties()
        res.append(('object', self.object))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.copy import GNCopyChildController
        return GNCopyChildController(document, model, self)


class GNCDelete(GNCopyChild):

    def tag_name(self, full_name=True):
        return "delete"

    def get_controller(self, document, model):
        from ...controller.geometry.copy import GNCDeleteController
        return GNCDeleteController(document, model, self)


class GNCReplace(GNCopyChild):

    def __init__(self, parent=None, object=None, replacer=None):
        super(GNCReplace, self).__init__(parent, object)
        self.replacer = replacer    # with in PLaSK

    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNCReplace, self)._attributes_from_xml(attribute_reader, conf)
        self.replacer = attribute_reader.get('with')

    def _attributes_to_xml(self, element, conf):
        super(GNCReplace, self)._attributes_to_xml(element, conf)
        if self.replacer is not None: element.attrib['with'] = self.replacer

    def _children_from_xml(self, ordered_reader, conf):
        if self.replacer is None:
            construct_geometry_object(ordered_reader.require(), conf)

    def accept_new_child(self):
        return not self.children

    def tag_name(self, full_name=True):
        return "replace"

    def major_properties(self):
        res = super(GNCReplace, self).major_properties()
        res.append(('with', self.replacer))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.copy import GNCReplaceController
        return GNCReplaceController(document, model, self)


class GNCToBlock(GNCopyChild):

    def __init__(self, parent=None, object=None, material=None):
        super(GNCToBlock, self).__init__(parent, object)
        self.material = material    # with in PLaSK

    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNCToBlock, self)._attributes_from_xml(attribute_reader, conf)
        self.material = attribute_reader.get('material')

    def _attributes_to_xml(self, element, conf):
        super(GNCToBlock, self)._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'material')

    def tag_name(self, full_name=True):
        return "toblock"

    def major_properties(self):
        res = super(GNCToBlock, self).major_properties()
        res.append(('material', self.material))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.copy import GNCToBlockController
        return GNCToBlockController(document, model, self)


class GNCopy(GNObject):

    def __init__(self, parent=None, name=None, source=None):
        super(GNCopy, self).__init__(parent, name)
        self.source = source    # from in PLaSK

    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNCopy, self)._attributes_from_xml(attribute_reader, conf)
        self.source = attribute_reader.get('from')

    def _attributes_to_xml(self, element, conf):
        super(GNCopy, self)._attributes_to_xml(element, conf)
        if self.source is not None: element.attrib['from'] = self.source

    def _children_from_xml(self, ordered_reader, conf):
        for t in ordered_reader.iter():
            if t.tag == 'delete': el = GNCDelete(parent=self)
            elif t.tag == 'replace': el = GNCReplace(parent=self)
            elif t.tag == 'toblock': el = GNCToBlock(parent=self)
            else: ordered_reader.recent_was_unexpected()
            el.set_xml_element(t, conf)

    def tag_name(self, full_name=True):
        return "copy"

    def stub(self):
        if self.name is not None and '{' not in self.name and self.source is not None and '{' not in self.source:
            return '    {} = {}'.format(self.name, self.source)
        return ''

    def accept_new_child(self):
        return True

    def add_child_options(self):
        return [{
                'delete': lambda i1, i2: GNCDelete(),
                'replace': lambda i1, i2: GNCReplace(),
                'toblock': lambda i1, i2: GNCToBlock()
                }]

    def get_controller(self, document, model):
        from ...controller.geometry.copy import GNCopyController
        return GNCopyController(document, model, self)

    @classmethod
    def from_xml(cls, element, conf):
        result = GNCopy()
        result.set_xml_element(element, conf)
        return result