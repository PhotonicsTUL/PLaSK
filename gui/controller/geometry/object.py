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
from ..defines import get_defines_completer
from ...qt import QtGui
from ...utils.str import empty_to_none, none_to_empty


class GNObjectController(GNodeController):

    def fill_form(self):
        self.role = QtGui.QLineEdit()
        self.role.setCompleter(self.defines_completer)
        self.form_layout.addRow('roles', self.role)

    def save_data_in_model(self):
        self.node.role = empty_to_none(self.role.text())

    def on_edit_enter(self):
        self.role.setText(none_to_empty(self.node.role))
        pass
        #self.notify_changes = False
        #for i in range(0, self.model.dim):
        #    self.axis_edit[i].from_model(self.model.axis[i])
        #self.notify_changes = True
