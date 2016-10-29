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
from ...utils.str import none_to_empty


class GNAgainController(GNodeController):

    def construct_form(self):
        super(GNAgainController, self).construct_form()
        self.construct_group('Again Settings')
        self.ref = self.construct_names_before_self_combo_box('Referenced object:', node_property_name='ref')
        self.ref.setToolTip('&lt;again <b>ref</b>=""/&gt;<br/>'
                            'Name of the referenced object.')
        self.in_parent_controller = self.node.get_controller_for_inparent(self.document, self.model)
        if self.in_parent_controller is not None:
            self.vbox.insertWidget(0, self.in_parent_controller.get_widget())

    def fill_form(self):
        super(GNAgainController, self).fill_form()
        with BlockQtSignals(self.ref):
            self.ref.setEditText(none_to_empty(self.node.ref))


class GNCopyChildController(GNodeController):

    def construct_form(self):
        super(GNCopyChildController, self).construct_form()
        self.construct_group('Operation Settings')
        self.object = self.construct_names_before_self_combo_box('Object:', node_property_name='object')

    def fill_form(self):
        super(GNCopyChildController, self).fill_form()
        with BlockQtSignals(self.object):
            self.object.setEditText(none_to_empty(self.node.object))


class GNCDeleteController(GNCopyChildController):

    def construct_form(self):
        super(GNCDeleteController, self).construct_form()
        self.object.setToolTip('&lt;delete <b>object</b>=""/&gt;<br/>'
                                'Name of the object to delete. Required.')


class GNCReplaceController(GNCopyChildController):

    def construct_form(self):
        super(GNCReplaceController, self).construct_form()
        self.object.setToolTip('&lt;replace <b>object</b>="" with=""/&gt;<br/>'
                                'Name of the object to delete (replace). Required.')
        self.replacer = self.construct_names_before_self_combo_box('With:',
                          node_property_name='replacer', display_property_name='name of the object to replace with')
        self.replacer.setToolTip('&lt;replace object="" <b>with</b>=""/&gt;<br/>'
            'Name of the object to replace with. This object does not need to be located in the subtree of the copied object.')

    def fill_form(self):
        super(GNCReplaceController, self).fill_form()
        with BlockQtSignals(self.replacer):
            self.replacer.setEditText(none_to_empty(self.node.replacer))


class GNCToBlockController(GNCopyChildController):

    def construct_form(self):
        super(GNCToBlockController, self).construct_form()
        self.object.setToolTip('&lt;toblock <b>object</b>="" material=""/&gt;<br/>'
                               'Name of the object to replace with the the solid block. Required.')
        self.material = self.construct_material_combo_box('Block material:', items=[''], node_property_name='material')
        self.material.setToolTip('&lt;toblock object="" <b>material</b>=""/&gt;<br/>'
                                 'Material of the solid block. Required.')
        self.name = self.construct_line_edit('Name:', node_property_name='name')
        self.name.setToolTip('&lt;{} <b>name</b>="" ...&gt;<br/>'
                             'Replacing block name for further reference.'
                             ' In the script section, the object is available by GEO table,'
                             ' which is indexed by names of geometry objects.'.format(self.node.tag_name(False)))
        self.role = self.construct_line_edit('Roles:', node_property_name='role', display_property_name='roles')
        self.role.setToolTip('&lt;{} <b>role</b>="" ...&gt;<br/>'
                             'Replacing block role. Important for some solvers.'.format(self.node.tag_name(False)))

    def fill_form(self):
        super(GNCToBlockController, self).fill_form()
        with BlockQtSignals(self.material):
            self.material.setEditText(none_to_empty(self.node.material))
            self.name.setText(none_to_empty(self.node.name))
            self.role.setText(none_to_empty(self.node.role))


class GNCopyController(GNObjectController):

    def construct_form(self):
        self.construct_group('Copy Settings')
        self.source = self.construct_names_before_self_combo_box('From:', node_property_name='source')
        self.source.setToolTip('&lt;copy <b>from</b>="" ...&gt;<br/>'
                                'Name of the source two or three dimensional object to make modified copy of.'
                                ' Usually it is some container that has some other named its items or sub-items.'
                                ' Required.')
        super(GNCopyController, self).construct_form()

    def fill_form(self):
        super(GNCopyController, self).fill_form()
        with BlockQtSignals(self.source):
            self.source.setEditText(none_to_empty(self.node.source))