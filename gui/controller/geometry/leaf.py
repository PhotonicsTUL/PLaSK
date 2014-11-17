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

from ...qt import QtGui

from .object import GNObjectController
from ...utils.str import empty_to_none, none_to_empty

class GNLeafController(GNObjectController):

    def fill_form(self):
        #TODO material
        self.construct_group('Mesh division settings')
        self.step_num = self.construct_line_edit('maximum number of steps')
        self.step_dist = self.construct_line_edit('minimum step size')
        super(GNLeafController, self).fill_form()

    def save_data_in_model(self):
        super(GNLeafController, self).save_data_in_model()
        self.node.step_num = empty_to_none(self.step_num.text())
        self.node.step_dist = empty_to_none(self.step_dist.text())

    def on_edit_enter(self):
        super(GNLeafController, self).on_edit_enter()
        self.step_num.setText(none_to_empty(self.node.step_num))
        self.step_dist.setText(none_to_empty(self.node.step_dist))


class GNBlockController(GNLeafController):

    def fill_form(self):
        self.construct_group('Block-specific settings')
        self.size = self.construct_point_controllers(row_name='size')
        super(GNBlockController, self).fill_form()

    def save_data_in_model(self):
        super(GNBlockController, self).save_data_in_model()
        self.node.size = [empty_to_none(p.text()) for p in self.size]

    def on_edit_enter(self):
        super(GNBlockController, self).on_edit_enter()
        for i in range(0, self.node.dim):
            self.size[i].setText(none_to_empty(self.node.size[i]))


class GNTriangleController(GNLeafController):

    def fill_form(self):
        self.construct_group('Vertexes coordinates (other than: {}):'.format(', '.join('0' for _ in range(0, self.node.dim))))
        self.p0 = self.construct_point_controllers(row_name='first')
        self.p1 = self.construct_point_controllers(row_name='second')
        super(GNTriangleController, self).fill_form()

    def save_data_in_model(self):
        super(GNTriangleController, self).save_data_in_model()
        self.node.points = (
            tuple(empty_to_none(p.text()) for p in self.p0),
            tuple(empty_to_none(p.text()) for p in self.p1)
        )

    def on_edit_enter(self):
        super(GNTriangleController, self).on_edit_enter()
        for i in range(0, self.node.dim):
            self.p0[i].setText(none_to_empty(self.node.points[0][i]))
            self.p1[i].setText(none_to_empty(self.node.points[1][i]))


class GNCircleController(GNLeafController):

    def fill_form(self):
        self.construct_group('{} size:'.format('Circle' if self.node.dim == 2 else 'Sphere'))
        self.radius = self.construct_line_edit('radius')
        super(GNCircleController, self).fill_form()

    def save_data_in_model(self):
        super(GNCircleController, self).save_data_in_model()
        self.node.radius = empty_to_none(self.radius.text())

    def on_edit_enter(self):
        super(GNCircleController, self).on_edit_enter()
        self.radius.setText(none_to_empty(self.node.radius))



class GNCylinderController(GNLeafController):

    def fill_form(self):
        self.construct_group('Cylinder size:')
        self.radius = self.construct_line_edit('radius')
        self.height = self.construct_line_edit('height')
        super(GNCylinderController, self).fill_form()

    def save_data_in_model(self):
        super(GNCylinderController, self).save_data_in_model()
        self.node.radius = empty_to_none(self.radius.text())
        self.node.height = empty_to_none(self.height.text())

    def on_edit_enter(self):
        super(GNCylinderController, self).on_edit_enter()
        self.radius.setText(none_to_empty(self.node.radius))
        self.height.setText(none_to_empty(self.node.height))
