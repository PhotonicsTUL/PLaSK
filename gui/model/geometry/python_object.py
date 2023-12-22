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

# Copyright (C) 2021 Photonics Group, Lodz University of Technology
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

import weakref
from lxml import etree

from .object import GNObject
from .reader import GNReadConf
from ...utils.xml import get_text_unindent, make_indented_text, AttributeReader, xml_to_attr, attr_to_xml


class GNPython(GNObject):

    have_mesh_settings = False

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.code = "return ..."

    def tag_name(self, full_name=True):
        return 'python'

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
        self.code = get_text_unindent(element)
        subtree_conf.after_read(self)

    def make_xml_element(self, conf):
        """
        Get XML representation of self.
        Use _attributes_to_xml and get_child_xml_element.
        :param GNReadConf conf: reader configuration
        :return etree.Element: XML representation of self (with children)
        """
        subtree_conf = GNReadConf(conf)
        self.preset_conf(subtree_conf)
        res = etree.Element(self.tag_name())
        self._attributes_to_xml(res, subtree_conf)
        if self.code is not None:
            res.text = make_indented_text(self.code, sum(1 for _ in self.path_to_root))
        return res

    def stub(self):
        if self.name is not None and '{' not in self.name:
            return '    {} = {}'.format(self.name.replace('-', '_'), self.python_type())
        return ''

    # def create_info(self, res, names):
    #     super().create_info(res, names)

    def get_controller(self, document, model):
        from ...controller.geometry.python_object import GNPythonController
        return GNPythonController(document, model, self)

    @staticmethod
    def from_xml(element, conf):
        result = GNPython()
        result.load_xml_element(element, conf)
        return result
