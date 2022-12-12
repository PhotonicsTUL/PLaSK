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

from ..qt.QtWidgets import *
from ..qt.QtGui import *


class UndoCommandWithSetter(QUndoCommand):
    """
        Undo command which change node using setter method, to new_value and call model.fire_changed after each change.
        Node can optionally be equal to model.
    """

    def __init__(self, model, setter, new_value, old_value, action_name, parent=None):
        super().__init__(action_name, parent)
        self.model = model
        self.setter = setter
        self.new_value = new_value
        self.old_value = old_value

    def set_property_value(self, value):
        self.setter(value)
        self.model.fire_changed()

    def redo(self):
        self.set_property_value(self.new_value)

    def undo(self):
        self.set_property_value(self.old_value)
