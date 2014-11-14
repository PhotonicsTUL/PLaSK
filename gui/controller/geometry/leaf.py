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
    pass    #TODO


class GNBlockController(GNLeafController):

    def fill_form(self):
        self.construct_group('Block-specific settings')
        self.size = self.construct_point_controllers(row_name='size')
        super(GNLeafController, self).fill_form()

    def save_data_in_model(self):
        super(GNBlockController, self).save_data_in_model()
        self.node.size = [empty_to_none(p.text()) for p in self.size]

    def on_edit_enter(self):
        super(GNBlockController, self).on_edit_enter()
        for i in range(0, self.node.dim):
            self.size[i].setText(none_to_empty(self.node.size[i]))


class GNTriangleController(GNLeafController):

    def fill_form(self):
        self.construct_group('Vertexes coordinates:')
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