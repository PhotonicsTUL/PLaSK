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

from .node import GNode
from .reader import axes_as_list
from ...utils.xml import xml_to_attr, attr_to_xml


class GNObject(GNode):
    """Base class for all nodes read by GeometryReader::readObject() in PLaSK."""

    #next_serial_nr = 0

    def __init__(self, parent=None, dim=None, children_dim=None):
        super(GNObject, self).__init__(parent, dim, children_dim)
        self.name = None
        self.role = None
        self.axes = None
        #self.serial_nr = GNObject.next_serial_nr
        #GNObject.next_serial_nr += 1

    #def serial_name(self):
    #    return '#GUI{}'.format(self.serial_nr)

    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNObject, self)._attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'name', 'role', 'axes')
        self.step_num = attribute_reader.get('steps-num')
        self.step_dist = attribute_reader.get('steps-dist')
        if self.axes is not None: conf.axes = self.axes

    def _attributes_to_xml(self, element, conf):
        super(GNObject, self)._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'name', 'role', 'axes')
        if self.step_num is not None: element.attrib['steps-num'] = self.step_num
        if self.step_dist is not None: element.attrib['steps-dist'] = self.step_dist
        if self.axes is not None: conf.axes = self.axes

    def python_type(self):
        return 'None'

    def stub(self):
        if self.name is None or '{' in self.name:
            res = ''
        else:
            res = '    {} = {}()\n'.format(self.name.replace('-', '_'), self.python_type())
        for c in self.children:
            cs = c.stub()
            if cs:
                res += cs
                res += '\n'
        return res[:-1]

    def get_controller(self, document, model):
        from ...controller.geometry.object import GNObjectController
        return GNObjectController(document, model, self)

    def minor_properties(self):
        res = super(GNObject, self).minor_properties()
        res.append(('axes', self.axes))
        res.append(('role', self.role))
        return res
    
    def get_axes_conf(self):
        if self.axes is not None: return axes_as_list(self.axes)
        return super(GNObject, self).get_axes_conf()

    def create_info(self, res, names):
        super(GNObject, self).create_info(res, names)
        if self.name is not None:
            names.setdefault(self.name, []).append(self)