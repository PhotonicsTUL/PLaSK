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
from plask import material

from .object import GNObjectController
from .node import GNodeController
from ...utils.str import empty_to_none, none_to_empty


class GNAgainController(GNodeController):

    def fill_form(self):
        super(GNAgainController, self).fill_form()
        self.ref = self.construct_line_edit('ref')

    def save_data_in_model(self):
        super(GNAgainController, self).save_data_in_model()
        self.node.ref = empty_to_none(self.ref.text())

    def on_edit_enter(self):
        super(GNAgainController, self).on_edit_enter()
        self.ref.setText(none_to_empty(self.node.ref))


class GNCopyChildController(GNodeController):

    def fill_form(self):
        super(GNCopyChildController, self).fill_form()
        self.object = self.construct_line_edit('object')

    def save_data_in_model(self):
        super(GNCopyChildController, self).save_data_in_model()
        self.node.object = empty_to_none(self.object.text())

    def on_edit_enter(self):
        super(GNCopyChildController, self).on_edit_enter()
        self.object.setText(none_to_empty(self.node.object))


class GNCReplaceController(GNodeController):

    def fill_form(self):
        super(GNCReplaceController, self).fill_form()
        self.replacer = self.construct_line_edit('with')

    def save_data_in_model(self):
        super(GNCReplaceController, self).save_data_in_model()
        self.node.replacer = empty_to_none(self.replacer.text())

    def on_edit_enter(self):
        super(GNCReplaceController, self).on_edit_enter()
        self.replacer.setText(none_to_empty(self.node.replacer))


class GNCToBlockController(GNodeController):

    def fill_form(self):
        super(GNCToBlockController, self).fill_form()
        self.material = self.construct_line_edit('material')

    def save_data_in_model(self):
        super(GNCToBlockController, self).save_data_in_model()
        self.node.material = empty_to_none(self.material.text())

    def on_edit_enter(self):
        super(GNCToBlockController, self).on_edit_enter()
        self.material.setText(none_to_empty(self.node.material))


class GNCopyController(GNObjectController):

    def fill_form(self):
        super(GNCopyController, self).fill_form()
        self.source = self.construct_line_edit('from')

    def save_data_in_model(self):
        super(GNCopyController, self).save_data_in_model()
        self.node.source = empty_to_none(self.source.text())

    def on_edit_enter(self):
        super(GNCopyController, self).on_edit_enter()
        self.source.setText(none_to_empty(self.node.source))