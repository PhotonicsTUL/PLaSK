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
from .node import GNodeController
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty


class GNAgainController(GNodeController):

    def fill_form(self):
        super(GNAgainController, self).fill_form()
        self.ref = self.construct_names_before_self_combo_box('ref')

    def save_data_in_model(self):
        super(GNAgainController, self).save_data_in_model()
        self.node.ref = empty_to_none(self.ref.currentText())

    def on_edit_enter(self):
        super(GNAgainController, self).on_edit_enter()
        with BlockQtSignals(self.ref) as ignored:
            self.ref.setEditText(none_to_empty(self.node.ref))


class GNCopyChildController(GNodeController):

    def fill_form(self):
        super(GNCopyChildController, self).fill_form()
        self.object = self.construct_names_before_self_combo_box('object')

    def save_data_in_model(self):
        super(GNCopyChildController, self).save_data_in_model()
        self.node.object = empty_to_none(self.object.currentText())

    def on_edit_enter(self):
        super(GNCopyChildController, self).on_edit_enter()
        with BlockQtSignals(self.object) as _:
            self.object.setEditText(none_to_empty(self.node.object))


class GNCReplaceController(GNCopyChildController):

    def fill_form(self):
        super(GNCReplaceController, self).fill_form()
        self.replacer = self.construct_names_before_self_combo_box('with')

    def save_data_in_model(self):
        super(GNCReplaceController, self).save_data_in_model()
        self.node.replacer = empty_to_none(self.replacer.currentText())

    def on_edit_enter(self):
        super(GNCReplaceController, self).on_edit_enter()
        with BlockQtSignals(self.replacer) as _:
            self.replacer.setEditText(none_to_empty(self.node.replacer))


class GNCToBlockController(GNCopyChildController):

    def fill_form(self):
        super(GNCToBlockController, self).fill_form()
        self.material = self.construct_material_combo_box('block material', items=[''])

    def save_data_in_model(self):
        super(GNCToBlockController, self).save_data_in_model()
        self.node.material = empty_to_none(self.material.currentText())

    def on_edit_enter(self):
        super(GNCToBlockController, self).on_edit_enter()
        with BlockQtSignals(self.material) as _:
            self.material.setEditText(none_to_empty(self.node.material))


class GNCopyController(GNObjectController):

    def fill_form(self):
        self.construct_group('Copy-specific settings')
        self.source = self.construct_names_before_self_combo_box('from')
        super(GNCopyController, self).fill_form()

    def save_data_in_model(self):
        super(GNCopyController, self).save_data_in_model()
        self.node.source = empty_to_none(self.source.currentText())

    def on_edit_enter(self):
        super(GNCopyController, self).on_edit_enter()
        with BlockQtSignals(self.source) as _:
            self.source.setEditText(none_to_empty(self.node.source))