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

from .object import GNObject
from ...utils.xml import xml_to_attr, attr_to_xml


class GNLeaf(GNObject):

    def __init__(self, parent=None, dim=None):
        super(GNLeaf, self).__init__(parent=parent, dim=dim, children_dim=None)
        self.step_num = None
        self.step_dist = None
        self.material_top = None
        self.material_bottom = None
        
    def attributes_from_xml(self, attribute_reader, conf):
        super(GNLeaf, self).attributes_from_xml(attribute_reader, conf)
        self.step_num = attribute_reader.get('step-num')
        self.step_dist = attribute_reader.get('step-dist')
        self.set_material(attribute_reader.get('material'))
        if self.material_bottom is None:
            self.material_bottom = attribute_reader.get('material-bottom')
            self.material_top = attribute_reader.get('material-top')

    def attributes_to_xml(self, element, conf):
        super(GNLeaf, self).attributes_to_xml(element, conf)
        if self.step_num is not None: element.attrib['step-num'] = self.step_num
        if self.step_dist is not None: element.attrib['step-dist'] = self.step_dist
        if self.material_top == self.material_bottom:
            if self.material_top is not None: element.attrib['material'] = self.material_top
        else:
            if self.material_top is not None: element.attrib['material-top'] = self.material_top
            if self.material_bottom is not None: element.attrib['material-bottom'] = self.material_bottom

    def major_properties(self):
        res = super(GNLeaf, self).major_properties()
        if self.material_top == self.material_bottom:
            res.append(('material', self.material_top))
        else:
            if self.material_top is not None and self.material_bottom is not None:
                res.append(('bottom / top materials', '{} / {}'.format(self.material_bottom, self.material_top)))
            else:
                res.append(('top material', self.material_top))
                res.append(('bottom material', self.material_bottom))
        return res


    def minor_properties(self):
        res = super(GNLeaf, self).minor_properties()
        res.append(('step dist', self.step_dist))
        res.append(('step num', self.step_num))
        return res

    def set_material(self, material):
        self.material_bottom = self.material_top = material

    def is_solid(self):
        return self.material_bottom == self.material_top

    def stub(self):
        return ''

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNLeafController
        return GNLeafController(document, model, self)


class GNBlock(GNLeaf):

    def __init__(self, parent=None, dim=None):
        super(GNBlock, self).__init__(parent=parent, dim=dim)
        self.size = [None for _ in range(0, dim)]

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNBlock, self).attributes_from_xml(attribute_reader, conf)
        self.size = [attribute_reader.get('d'+a) for a in conf.axes_names(self.dim)]

    def attributes_to_xml(self, element, conf):
        super(GNBlock, self).attributes_to_xml(element, conf)
        for a in range(0, self.dim):
            if self.size[a] is not None:
                element.attrib['d'+conf.axis_name(self.dim, a)] = self.size[a]

    def tag_name(self, full_name=True):
        return "block{}d".format(self.dim) if full_name else "block"

    def python_type(self):
        return 'geometry.Block{}D'.format(self.dim)

    def major_properties(self):
        res = super(GNBlock, self).major_properties()
        if any(self.size): res.append(('size', ', '.join('?' if x is None else x for x in self.size)))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNBlockController
        return GNBlockController(document, model, self)

    @classmethod
    def from_xml_2d(cls, element, conf):
        result = GNBlock(dim=2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(cls, element, conf):
        result = GNBlock(dim=3)
        result.set_xml_element(element, conf)
        return result


class GNCylinder(GNLeaf):
    
    def __init__(self, parent=None):
        super(GNCylinder, self).__init__(parent=parent, dim=3)
        self.radius = None  #required in PLaSK but not in GUI
        self.height = None  #required in PLaSK but not in GUI

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNCylinder, self).attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'radius', 'height')

    def attributes_to_xml(self, element, conf):
        super(GNCylinder, self).attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'radius', 'height')

    def tag_name(self, full_name=True):
        return "cylinder"

    def python_type(self):
        return 'geometry.Cylinder'

    def major_properties(self):
        res = super(GNCylinder, self).major_properties()
        res.append(('radius', self.radius))
        res.append(('height', self.height))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNCylinderController
        return GNCylinderController(document, model, self)

    @classmethod
    def from_xml_3d(cls, element, conf):
        result = GNCylinder()
        result.set_xml_element(element, conf)
        return result
    

class GNCircle(GNLeaf):
    
    def __init__(self, parent=None, dim=None):
        super(GNCircle, self).__init__(parent=parent, dim=dim)
        self.radius = None  #required in PLaSK but not in GUI

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNCircle, self).attributes_from_xml(attribute_reader, conf)
        self.radius = attribute_reader.get('radius')

    def attributes_to_xml(self, element, conf):
        super(GNCircle, self).attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'radius')

    def tag_name(self, full_name=True):
        return "circle{}d".format(self.dim) if full_name else "circle"

    def python_type(self):
        return 'geometry.Circle' if self.dim == 2 else 'geometry.Sphere'

    def major_properties(self):
        res = super(GNCircle, self).major_properties()
        res.append(('radius', self.radius))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNCircleController
        return GNCircleController(document, model, self)

    @classmethod
    def from_xml_2d(cls, element, conf):
        result = GNCircle(dim=2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(cls, element, conf):
        result = GNCircle(dim=3)
        result.set_xml_element(element, conf)
        return result


class GNTriangle(GNLeaf):

    def __init__(self, parent=None):
        super(GNTriangle, self).__init__(parent=parent, dim=2)
        self.points = ((None, None), (None, None))

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNTriangle, self).attributes_from_xml(attribute_reader, conf)
        n = conf.axes_names(2)
        r = attribute_reader
        self.points = ((r.get(n[0] + '0'), r.get(n[1] + '0')), (r.get(n[0] + '1'), r.get(n[1] + '1')))

    def attributes_to_xml(self, element, conf):
        super(GNTriangle, self).attributes_to_xml(element, conf)
        axis_names = conf.axes_names(2)
        for point_nr in range(0, 2):
            for axis_nr in range(0, self.dim):
                v = self.points[point_nr][axis_nr]
                if v is not None: element.attrib[axis_names[axis_nr]] = v

    def tag_name(self, full_name=True):
        return "triangle"

    def python_type(self):
        return 'geometry.Triangle'

    def major_properties(self):
        res = super(GNTriangle, self).major_properties()
        points_str = ', '.join('({}, {})'.format(x[0] if x[0] else '?', x[1] if x[1] else '?') for x in self.points if x != (None, None))
        if points_str: res.append(('points', points_str))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNTriangleController
        return GNTriangleController(document, model, self)

    @classmethod
    def from_xml_2d(cls, element, conf):
        result = GNTriangle()
        result.set_xml_element(element, conf)
        return result
