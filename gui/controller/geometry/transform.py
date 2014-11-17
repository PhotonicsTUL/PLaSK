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

from .object import GNObjectController
from ...utils.str import empty_to_none, none_to_empty


class GNClipController(GNObjectController):

    def fill_form(self):
        self.construct_group('Clipping box')
        for b in self.node.bound_names():
            setattr(self, b, self.construct_line_edit(b))
        super(GNClipController, self).fill_form()

    def save_data_in_model(self):
        super(GNClipController, self).save_data_in_model()
        for b in self.node.bound_names():
            setattr(self.node, b, empty_to_none(getattr(self, b).text()))

    def on_edit_enter(self):
        super(GNClipController, self).on_edit_enter()
        for b in self.node.bound_names():
            getattr(self, b).setText(none_to_empty(getattr(self.node, b)))
