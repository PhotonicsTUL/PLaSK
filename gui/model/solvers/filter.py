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
from utils.xml import AttributeReader
from . import Solver


class FilterSolver(Solver):

    def __init__(self, what='', name='', parent=None, info_cb=None):
        super(FilterSolver, self).__init__('filter', parent=parent, info_cb=info_cb)
        self.what = what
        self.name = name
        self.geometry = ''

    def get_xml_element(self):
        return etree.Element(self.category, {"name": self.name, "for": self.what, "geometry": self.geometry})

    def set_xml_element(self, element):
        self.category = element.tag
        with AttributeReader(element) as attr:
            self.name = attr.get('name', None)
            self.what = attr.get('for', None)
            self.geometry = attr.get('geometry', None)

    def get_controller(self, document):
        from ...controller.solvers import FilterController
        return FilterController(document, self)

    def stub(self):
        return "{} = flow.{}Filter()".format(self.name, self.what)  # TODO: Geometry suffix