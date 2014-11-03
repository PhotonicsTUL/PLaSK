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
from ..defines import get_defines_completer
from ...qt import QtGui
from ...utils.str import empty_to_none, none_to_empty


class GNObjectController(GNodeController):

    def fill_form(self):
        self.role = QtGui.QLineEdit()
        self.role.setCompleter(self.defines_completer)
        self.form_layout.addRow('roles', self.role)

        self.axes = QtGui.QComboBox()
        self.axes.setEditable(True)     #TODO zablokować możliwość podawania {
        self.axes.addItems(['', 'x, y, z', 'z, x, y', 'p, r, z', 'l, t, v', 'long, tran, vert'])
        self.form_layout.addRow('axes', self.axes)

    def save_data_in_model(self):
        self.node.role = empty_to_none(self.role.text())
        self.node.axes = empty_to_none(self.axes.currentText())

    def on_edit_enter(self):
        self.role.setText(none_to_empty(self.node.role))
        self.axes.setEditText(axes_to_str(self.node.axes))
