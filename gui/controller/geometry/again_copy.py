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
from .object import GNObjectController
from .node import GNodeController
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty


class GNAgainController(GNodeController):

    def fill_form(self):
        super(GNAgainController, self).fill_form()
        self.construct_group('Again Settings')
        self.ref = self.construct_names_before_self_combo_box('Referenced object:', node_property_name='ref')
        self.ref.setToolTip('&lt;again <b>ref</b>=""/&gt;<br/>'
                            'Name of the referenced object.')
        self.in_parent_controller = self.node.get_controller_for_inparent(self.document, self.model)
        if self.in_parent_controller is not None:
            self.vbox.insertWidget(0, self.in_parent_controller.get_widget())

    #def save_data_in_model(self):
        #super(GNAgainController, self).save_data_in_model()
        #self.node.ref = empty_to_none(self.ref.currentText())

    def on_edit_enter(self):
        super(GNAgainController, self).on_edit_enter()
        with BlockQtSignals(self.ref) as _:
            self.ref.setEditText(none_to_empty(self.node.ref))


class GNCopyChildController(GNodeController):

    def fill_form(self):
        super(GNCopyChildController, self).fill_form()
        self.construct_group('Operation Settings')
        self.object = self.construct_names_before_self_combo_box('Object:', node_property_name='object')

    #def save_data_in_model(self):
        #super(GNCopyChildController, self).save_data_in_model()
        #self.node.object = empty_to_none(self.object.currentText())

    def on_edit_enter(self):
        super(GNCopyChildController, self).on_edit_enter()
        with BlockQtSignals(self.object) as _:
            self.object.setEditText(none_to_empty(self.node.object))


class GNCDeleteController(GNCopyChildController):

    def fill_form(self):
        super(GNCDeleteController, self).fill_form()
        self.object.setToolTip('&lt;delete <b>object</b>=""/&gt;<br/>'
                                'Name of the object to delete. Required.')


class GNCReplaceController(GNCopyChildController):

    def fill_form(self):
        super(GNCReplaceController, self).fill_form()
        self.object.setToolTip('&lt;replace <b>object</b>="" with=""/&gt;<br/>'
                                'Name of the object to delete (replace). Required.')
        self.replacer = self.construct_names_before_self_combo_box('With:',
                          node_property_name='replacer', display_property_name='name of the object to replace with')
        self.replacer.setToolTip('&lt;replace object="" <b>with</b>=""/&gt;<br/>'
            'Name of the object to replace with. This object does not need to be located in the subtree of the copied object.')

    #def save_data_in_model(self):
        #super(GNCReplaceController, self).save_data_in_model()
        #self.node.replacer = empty_to_none(self.replacer.currentText())

    def on_edit_enter(self):
        super(GNCReplaceController, self).on_edit_enter()
        with BlockQtSignals(self.replacer) as _:
            self.replacer.setEditText(none_to_empty(self.node.replacer))


class GNCToBlockController(GNCopyChildController):

    def fill_form(self):
        super(GNCToBlockController, self).fill_form()
        self.object.setToolTip('&lt;toblock <b>object</b>="" material=""/&gt;<br/>'
                                'Name of the object to replace with the the solid block. Required.')
        self.material = self.construct_material_combo_box('block material', items=[''])
        self.material.setToolTip('&lt;toblock object="" <b>material</b>=""/&gt;<br/>'
                                'Material of the solid block. Required.')

    def save_data_in_model(self):
        super(GNCToBlockController, self).save_data_in_model()
        self.node.material = empty_to_none(self.material.currentText())

    def on_edit_enter(self):
        super(GNCToBlockController, self).on_edit_enter()
        with BlockQtSignals(self.material) as _:
            self.material.setEditText(none_to_empty(self.node.material))


class GNCopyController(GNObjectController):

    def fill_form(self):
        self.construct_group('Copy Settings')
        self.source = self.construct_names_before_self_combo_box('From:', node_property_name='source')
        self.source.setToolTip('&lt;copy <b>from</b>="" ...&gt;<br/>'
                                'Name of the source two or three dimensional object to make modified copy of.'
                                ' Usually it is some container that has some other named its items or sub-items.'
                                ' Required.')
        super(GNCopyController, self).fill_form()

    #def save_data_in_model(self):
        #super(GNCopyController, self).save_data_in_model()
        #self.node.source = empty_to_none(self.source.currentText())

    def on_edit_enter(self):
        super(GNCopyController, self).on_edit_enter()
        with BlockQtSignals(self.source) as _:
            self.source.setEditText(none_to_empty(self.node.source))