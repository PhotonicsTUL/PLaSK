# -*- coding: utf-8 -*-
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

from .node import GNodeController
from ...model.geometry.reader import axes_to_str
from ...utils.str import empty_to_none, none_to_empty


class GNObjectController(GNodeController):

    #def __init__(self, document, model, node):
    #    super(GNObjectController, self).__init__(document, model, node)
    #    self.in_parent_controller = self.node.get_controller_for_inparent()

    def fill_form(self):
        self.construct_group('Basic settings')
        self.name = self.construct_line_edit('name')
        self.role = self.construct_line_edit('roles')
        self.axes = self.construct_combo_box('axes', ['', 'x, y, z', 'z, x, y', 'p, r, z', 'l, t, v', 'long, tran, vert'])
        self.in_parent_controller = self.node.get_controller_for_inparent(self.document, self.model)
        if self.in_parent_controller is not None:
            self.vbox.insertWidget(0, self.in_parent_controller.get_widget())
            #self.construct_group('Position in container', position=0)
            #self._current_form.addChildWidget(in_parent_widget)

    def save_data_in_model(self):
        self.node.name = empty_to_none(self.name.text())
        self.node.role = empty_to_none(self.role.text())
        self.node.axes = empty_to_none(self.axes.currentText())
        if self.in_parent_controller is not None: self.in_parent_controller.save_data_in_model()

    def on_edit_enter(self):
        self.name.setText(none_to_empty(self.node.name))
        self.role.setText(none_to_empty(self.node.role))
        self.axes.setEditText(axes_to_str(self.node.axes))
        if self.in_parent_controller is not None: self.in_parent_controller.on_edit_enter()
