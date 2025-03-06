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

import re  # DEPRECATED

from .object import GNObject
from ...utils.validators import can_be_int, can_be_float, can_be_double_float, can_be_list
from ...utils.xml import xml_to_attr, attr_to_xml


class GNLeaf(GNObject):

    def __init__(self, parent=None, dim=None):
        super().__init__(parent=parent, dim=dim, children_dim=None)
        self.material_top = None
        self.material_bottom = None
        self.material_shape = None

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.set_material(attribute_reader.get('material'))
        if self.material_bottom is None:
            self.material_bottom = attribute_reader.get('material-bottom')
            self.material_top = attribute_reader.get('material-top')
            self.material_shape = attribute_reader.get('material-shape')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        if self.material_top == self.material_bottom:
            if self.material_top is not None: element.attrib['material'] = self.material_top
        else:
            if self.material_top is not None: element.attrib['material-top'] = self.material_top
            if self.material_bottom is not None: element.attrib['material-bottom'] = self.material_bottom
            if self.material_shape is not None: element.attrib['material-shape'] = self.material_shape

    def major_properties(self):
        res = super().major_properties()
        if self.material_top == self.material_bottom:
            res.append(('material', self.material_bottom))
        else:
            #if self.material_top is not None and self.material_bottom is not None:
            #    res.append(('bottom / top materials', '{} / {}'.format(self.material_bottom, self.material_top)))
            #else:
            res.append('materials:')
            res.append(('top', self.material_top))
            res.append(('bottom', self.material_bottom))
            if self.material_shape is not None: res.append(('shape', self.material_shape))
            res.append(None)
        return res

    def minor_properties(self):
        res = super().minor_properties()
        res.append(('step dist', self.step_dist))
        res.append(('step num', self.step_num))
        return res

    def set_material(self, material):
        self.material_bottom = self.material_top = material

    def is_solid(self):
        return self.material_bottom == self.material_top

    def stub(self):
        if self.name is not None and '{' not in self.name:
            return '    {} = {}'.format(self.name.replace('-', '_'), self.python_type())
        return ''

    def create_info(self, res, names):
        super().create_info(res, names)
        if not self.material_bottom or not self.material_top:
            if not self.material_bottom and not self.material_top:
                self._require(res, 'material')
            else:
                what = 'material_bottom' if not self.material_bottom else 'material_top'
                self._require(res, what, what.replace('_', ' '))
        if not can_be_int(self.step_num): self._wrong_type(res, 'integer', 'step_num', 'maximum number of the mesh steps')
        if not can_be_float(self.step_dist): self._wrong_type(res, 'float', 'step_dist', 'minimum step size')
        if not can_be_float(self.material_shape): self._wrong_type(res, 'float', 'material_shape', 'exponent for graded materials')

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNLeafController
        return GNLeafController(document, model, self)


class GNBlock(GNLeaf):

    alt_names = ['length', 'width', 'height']

    def __init__(self, parent=None, dim=None):
        super().__init__(parent=parent, dim=dim)
        self.size = [None for _ in range(0, dim)]
        self.angle = None

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.size = [attribute_reader.get('d'+a) for a in conf.axes_names(self.dim)]
        for i, alt in enumerate(GNBlock.alt_names[3-self.dim:]):
            if self.size[i] is None:
                self.size[i] = attribute_reader.get(alt)
        if self.dim == 3:
            self.angle = attribute_reader.get('angle')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        for a in range(0, self.dim):
            if self.size[a] is not None: element.attrib['d'+conf.axis_name(self.dim, a)] = self.size[a]
        if self.dim == 3 and self.angle is not None:
            element.attrib['angle'] = self.angle

    def tag_name(self, full_name=True):
        return ('rectangle', 'cuboid')[self.dim-2]
        # return "block{}d".format(self.dim) if full_name else "block"

    def python_type(self):
        return 'geometry.Block{}D'.format(self.dim)

    def major_properties(self):
        res = super().major_properties()
        if any(self.size): res.append(('size', ', '.join('?' if x is None else x for x in self.size)))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNBlockController
        return GNBlockController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        for i, v in enumerate(self.size):
            if not can_be_float(v, required=True): self._require(res, ('size', i), 'size component', type='float')
        # if None in self.size: self._require(res, 'size', indexes=(self.size.index(None),))

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNBlock(dim=2)
        result.load_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNBlock(dim=3)
        result.load_xml_element(element, conf)
        return result


class GNCylinder(GNLeaf):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=3)
        self.radius = None  #required in PLaSK but not in GUI
        self.height = None  #required in PLaSK but not in GUI

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'radius', 'height')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'radius', 'height')

    def tag_name(self, full_name=True):
        return "cylinder"

    def python_type(self):
        return 'geometry.Cylinder'

    def major_properties(self):
        res = super().major_properties()
        res.append(('radius', self.radius))
        res.append(('height', self.height))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNCylinderController
        return GNCylinderController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        if not can_be_float(self.radius, required=True): self._require(res, 'radius', type='float')
        if not can_be_float(self.height, required=True): self._require(res, 'height', type='float')

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNCylinder()
        result.load_xml_element(element, conf)
        return result


class GNTube(GNLeaf):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=3)
        self.inner_radius = None  #required in PLaSK but not in GUI
        self.outer_radius = None  #required in PLaSK but not in GUI
        self.height = None  #required in PLaSK but not in GUI

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        if attribute_reader is None:
            self.inner_radius = self.outer_radius = self.height = None
        else:
            self.inner_radius = attribute_reader.get('inner-radius')
            self.outer_radius = attribute_reader.get('outer-radius')
            self.height = attribute_reader.get('height')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        if self.inner_radius is not None: element.attrib['inner-radius'] = self.inner_radius
        if self.outer_radius is not None: element.attrib['outer-radius'] = self.outer_radius
        if self.height is not None: element.attrib['height'] = self.height

    def tag_name(self, full_name=True):
        return "tube"

    def python_type(self):
        return 'geometry.Tube'

    def major_properties(self):
        res = super().major_properties()
        res.append(('inner-radius', self.inner_radius))
        res.append(('outer-radius', self.outer_radius))
        res.append(('height', self.height))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNTubeController
        return GNTubeController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        if not can_be_float(self.inner_radius, required=True): self._require(res, 'inner_radius', 'inner-radius', type='float')
        if not can_be_float(self.outer_radius, required=True): self._require(res, 'outer_radius', 'outer-radius', type='float')
        if not can_be_float(self.height, required=True): self._require(res, 'height', type='float')

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNTube()
        result.load_xml_element(element, conf)
        return result


class GNEllipticCylinder(GNLeaf):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=3)
        self.radius0 = None
        self.radius1 = None
        self.angle = None
        self.height = None

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'radius0', 'radius1', 'angle', 'height')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'radius0', 'radius1', 'angle', 'height')

    def tag_name(self, full_name=True):
        return "elliptic-cylinder"

    def python_type(self):
        return 'geometry.EllipticCylinder'

    def major_properties(self):
        res = super().major_properties()
        res.append(('radius0', self.radius0))
        res.append(('radius1', self.radius1))
        if self.angle is not None:
            res.append(('angle', self.angle))
        res.append(('height', self.height))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNEllipticCylinderController
        return GNEllipticCylinderController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        if not can_be_float(self.radius0, required=True): self._require(res, 'radius0', type='float')
        if not can_be_float(self.radius1, required=True): self._require(res, 'radius1', type='float')
        if not can_be_float(self.angle): self._wrong_type(res, 'float', 'angle', 'rotation angle')
        if not can_be_float(self.height, required=True): self._require(res, 'height', type='float')

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNEllipticCylinder()
        result.load_xml_element(element, conf)
        return result


class GNCircle(GNLeaf):

    def __init__(self, parent=None, dim=None):
        super().__init__(parent=parent, dim=dim)
        self.radius = None  #required in PLaSK but not in GUI

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.radius = attribute_reader.get('radius')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'radius')

    def tag_name(self, full_name=True):
        return "sphere" if self.dim == 3 else "circle"

    def python_type(self):
        return 'geometry.Circle' if self.dim == 2 else 'geometry.Sphere'

    def major_properties(self):
        res = super().major_properties()
        res.append(('radius', self.radius))
        return res

    def create_info(self, res, names):
        super().create_info(res, names)
        if not can_be_float(self.radius, required=True): self._require(res, 'radius', type='float')

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNCircleController
        return GNCircleController(document, model, self)

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNCircle(dim=2)
        result.load_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNCircle(dim=3)
        result.load_xml_element(element, conf)
        return result


class GNEllipse(GNLeaf):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=2)
        self.radius0 = None
        self.radius1 = None

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.radius0 = attribute_reader.get('radius0')
        self.radius1 = attribute_reader.get('radius1')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'radius0')
        attr_to_xml(self, element, 'radius1')

    def tag_name(self, full_name=True):
        return "ellipse"

    def python_type(self):
        return 'geometry.Ellipse'

    def major_properties(self):
        res = super().major_properties()
        res.append(('radius0', self.radius0))
        res.append(('radius1', self.radius1))
        return res

    def create_info(self, res, names):
        super().create_info(res, names)
        if not can_be_float(self.radius0, required=True): self._require(res, 'radius0', type='float')
        if not can_be_float(self.radius1, required=True): self._require(res, 'radius1', type='float')

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNEllipseController
        return GNEllipseController(document, model, self)

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNEllipse()
        result.load_xml_element(element, conf)
        return result



class GNTriangle(GNLeaf):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=2)
        self.points = ((None, None), (None, None))

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        n = conf.axes_names(2)
        r = attribute_reader
        self.points = ((r.get('a' + n[0]), r.get('a' + n[1])), (r.get('b' + n[0]), r.get('b' + n[1])))

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        axis_names = conf.axes_names(2)
        for point_nr in range(0, 2):
            for axis_nr in range(0, self.dim):
                v = self.points[point_nr][axis_nr]
                if v is not None: element.attrib[('a', 'b')[point_nr] + axis_names[axis_nr]] = v

    def tag_name(self, full_name=True):
        return "triangle"

    def python_type(self):
        return 'geometry.Triangle'

    def major_properties(self):
        res = super().major_properties()
        points_str = ', '.join('({}, {})'.format(x[0] if x[0] else '?', x[1] if x[1] else '?')
                               for x in self.points if x != (None, None))
        if points_str: res.append(('points', points_str))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNTriangleController
        return GNTriangleController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        for p_nr in (0, 1):
            for i, v in enumerate(self.points[p_nr]):
                if not can_be_float(v, required=True):
                    self._require(res, ('points', p_nr, i), 'coordinate of the {} point'.format(('first', 'second')[p_nr]),
                                  type='float')
        #if None in self.points[0]: self._require(res, 'points', 'coordinate of first point', indexes=(0, self.points[0].index(None)))
        #if None in self.points[1]: self._require(res, 'points', 'coordinate of second point', indexes=(1, self.points[1].index(None)))

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNTriangle()
        result.load_xml_element(element, conf)
        return result


class GNTriangularPrism(GNLeaf):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=2)
        self.points = ((None, None), (None, None))
        self.height = None

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        n = conf.axes_names(3)[:2]
        r = attribute_reader
        self.points = ((r.get('a' + n[0]), r.get('a' + n[1])), (r.get('b' + n[0]), r.get('b' + n[1])))
        self.height = r.get('height')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        axis_names = conf.axes_names(3)[:2]
        for point_nr in range(0, 2):
            for axis_nr in range(0, self.dim):
                v = self.points[point_nr][axis_nr]
                if v is not None: element.attrib[('a', 'b')[point_nr] + axis_names[axis_nr]] = v
        attr_to_xml(self, element, 'height')

    def tag_name(self, full_name=True):
        return "triangular-prism"

    def python_type(self):
        return 'geometry.TriangularPrism'

    def major_properties(self):
        res = super().major_properties()
        points_str = ', '.join('({}, {})'.format(x[0] if x[0] else '?', x[1] if x[1] else '?')
                               for x in self.points if x != (None, None))
        if points_str:
            res.append(('base', points_str))
        if self.height is not None:
            res.append(('height', self.height))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNTriangularPrismController
        return GNTriangularPrismController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        for p_nr in (0, 1):
            for i, v in enumerate(self.points[p_nr]):
                if not can_be_float(v, required=True):
                    self._require(res, ('points', p_nr, i),
                                  'coordinate of the {} base point'.format(('first', 'second')[p_nr]),
                                  type='float')
        if not can_be_float(self.height, required=True):
            self._require(res, 'height', type='float')

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNTriangularPrism()
        result.load_xml_element(element, conf)
        return result


class GNPolygon(GNLeaf):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=2)
        self.vertices = ""

    def tag_name(self, full_name=True):
        return "polygon"

    def python_type(self):
        return 'geometry.Polygon'

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNPolygonController
        return GNPolygonController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        vertices = self.vertices.strip()
        if not can_be_list(vertices, required=True, item_validator=lambda v: can_be_double_float(v, required=True)):
            axes = self.get_axes_conf()
            self._require(res, 'vertices', f"list of vertices coordinates ({axes[-2]} {axes[-1]}) separated by semicolons")

    def make_xml_element(self, conf):
        el = super().make_xml_element(conf)
        el.text = self.vertices
        return el

    def _children_from_xml(self, ordered_reader, conf):
        self.vertices = ordered_reader.parent_element.text
        if self.vertices is None:
            self.vertices = ""
        self.vertices = self.vertices.strip()
        ordered_reader.require_end()

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNPolygon()
        result.load_xml_element(element, conf)
        return result


class GNPrism(GNLeaf):

    def __init__(self, parent=None):
        super().__init__(parent=parent, dim=2)
        self.height = None
        self.vertices = ""

    def tag_name(self, full_name=True):
        return "prism"

    def python_type(self):
        return 'geometry.Prism'

    def get_controller(self, document, model):
        from ...controller.geometry.leaf import GNPrismController
        return GNPrismController(document, model, self)

    def create_info(self, res, names):
        super().create_info(res, names)
        vertices = self.vertices.strip()
        if not can_be_list(vertices, required=True, item_validator=lambda v: can_be_double_float(v, required=True)):
            axes = self.get_axes_conf()
            self._require(res, 'vertices', f"list of lateral vertices coordinates ({axes[0]} {axes[1]}) separated by semicolons")

    def major_properties(self):
        res = super().major_properties()
        if self.height is not None:
            res.append(('height', self.height))
        return res

    def _attributes_from_xml(self, attribute_reader, conf):
        super()._attributes_from_xml(attribute_reader, conf)
        self.height = attribute_reader.get('height')

    def _attributes_to_xml(self, element, conf):
        super()._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'height')

    def make_xml_element(self, conf):
        el = super().make_xml_element(conf)
        el.text = self.vertices
        return el

    def _children_from_xml(self, ordered_reader, conf):
        self.vertices = ordered_reader.parent_element.text
        if self.vertices is None:
            self.vertices = ""
        self.vertices = self.vertices.strip()
        ordered_reader.require_end()

    OLD_ATTR_RE = re.compile(r'[ab][xyzrpltv]')  # DEPRECATED

    @staticmethod
    def from_xml_3d(element, conf):

        if element is not None:
            for a in element.attrib:
                # DEPRECATED
                if GNPrism.OLD_ATTR_RE.match(a):
                    result = GNTriangularPrism()
                    result.load_xml_element(element, conf)
                    return result

        result = GNPrism()
        result.load_xml_element(element, conf)
        return result
