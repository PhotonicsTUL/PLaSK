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

from ...utils.xml import OrderedTagReader, UnorderedTagReader, attr_to_xml, xml_to_attr, get_text
from ...utils.validators import can_be_float, can_be_int
from . import Grid

basestring = str, bytes
AXIS_NAMES = [
    [''], ['horizontal', 'vertical'], ['longitudinal', 'transverse', 'vertical']
]


class AxisConf:
    """Store axis configuration of rectilinear mesh"""

    def __init__(self, start=None, stop=None, num=None, points=None, type=None):
        self.start = start
        self.stop = stop
        self.num = num
        self.points = points
        self.type = None if type == '' else type
        self.comments = []

    def fill_xml_element(self, axis_element):
        attr_to_xml(self, axis_element, 'start', 'stop', 'num', 'type')
        if self.type != 'regular' and self.points:
            axis_element.text = self.points

    def set_from_xml(self, axis_element):
        xml_to_attr(axis_element, self, 'start', 'stop', 'num', 'type')
        self.points = None if axis_element is None else get_text(axis_element)

    def create_info(self, grid_model, res, rows, parent_property):
        if isinstance(parent_property, basestring): parent_property = (parent_property,)
        if not can_be_float(self.start):
            grid_model._required(res, rows, parent_property+('start',),
                                 '"start" (position of the first point on the axis)', type='float')
        if not can_be_float(self.stop):
            grid_model._required(res, rows, parent_property+('stop',),
                                 '"stop" (position of the last point on the axis)', type='float')
        if not can_be_int(self.num, int_validator=lambda n: n>0):
            grid_model._required(res, rows, parent_property+('num',),
                                 '"num" (number of the equally distributed points along the axis)', type='positive integer')


#RectangularMesh1D(Grid)
#if mesh_type in ('ordered', 'regular'):

class RectangularMesh1D(Grid):
    """Model of 1D rectangular mesh (ordered or regular)"""

    dim = 1

    @staticmethod
    def from_xml(grids_model, element):
        e = RectangularMesh1D(grids_model, element.attrib['name'], element.attrib['type'])
        e.load_xml_element(element)
        return e

    def __init__(self, grids_model, name, type):
        super(RectangularMesh1D, self).__init__(grids_model, name, type)
        self.axis = AxisConf()

    @property
    def is_regular(self):
        return self.type == 'regular'

    def make_xml_element(self):
        res = super(RectangularMesh1D, self).make_xml_element()
        for c in self.axis.comments:
            res.append(etree.Comment(c))
        self.axis.fill_xml_element(etree.SubElement(res, 'axis'))
        self.save_endcomments(res)
        return res

    def load_xml_element(self, element):
        super(RectangularMesh1D, self).load_xml_element(element)
        with OrderedTagReader(element) as r:
            el = r.get('axis')
            self.axis.set_from_xml(el)
            self.axis.comments = el.comments
            self.endcomments = r.get_comments()

    def get_controller(self, document):
        from ...controller.grids.mesh_rectilinear import RectangularMesh1DController
        return RectangularMesh1DController(document=document, model=self)

    def create_info(self, res, rows):
        super(RectangularMesh1D, self).create_info(res, rows)
        self.axis.create_info(self, res, rows, 'axis')


class RectangularMesh(Grid):
    """Model of 2D and 3D rectangular mesh"""

    @staticmethod
    def from_xml(grids_model, element):
        e = RectangularMesh(grids_model, int(element.attrib['type'][-2]), element.attrib['name'])
        e.load_xml_element(element)
        return e

    def __init__(self, grids_model, dim, name):
        super(RectangularMesh, self).__init__(grids_model, name, 'rectangular{}d'.format(dim))
        self.dim = dim
        self.axis = [AxisConf() for _ in range(0, self.dim)]

    @staticmethod
    def axis_tag_name(nr):
        return "axis{}".format(nr)

    def make_xml_element(self):
        res = super(RectangularMesh, self).make_xml_element()
        for i in range(0, self.dim):
            for c in self.axis[i].comments:
                res.append(etree.Comment(c))
            self.axis[i].fill_xml_element(etree.SubElement(res, RectangularMesh.axis_tag_name(i)))
        self.save_endcomments(res)
        return res

    def load_xml_element(self, element):
        super(RectangularMesh, self).load_xml_element(element)
        with UnorderedTagReader(element) as r:
            for i in range(0, self.dim):
                el = r.get(RectangularMesh.axis_tag_name(i))
                self.axis[i].set_from_xml(el)
                if el is not None:
                    self.axis[i].comments = el.comments
            self.endcomments = r.get_comments()

    def get_controller(self, document):
        from ...controller.grids.mesh_rectilinear import RectangularMeshController
        return RectangularMeshController(document=document, model=self)

    def create_info(self, res, rows):
        super(RectangularMesh, self).create_info(res, rows)
        for i in range(0, self.dim):
            self.axis[i].create_info(self, res, rows, ('axis', i))
