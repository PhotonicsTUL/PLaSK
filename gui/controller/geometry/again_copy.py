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

from .object import GNObjectController
from .node import GNodeController
from .leaf import GNMaterialControllerMixin
from ...utils.qsignals import BlockQtSignals
from ...utils.str import none_to_empty

from ...qt.QtWidgets import QSizePolicy


class GNAgainController(GNodeController):
    def construct_form(self):
        super().construct_form()
        self.construct_group('Again Settings')
        self.ref = self.construct_names_before_self_combo_box('Referenced object:', node_property_name='ref')
        self.ref.setToolTip('&lt;again <b>ref</b>=""/&gt;<br/>' 'Name of the referenced object.')
        self.in_parent_controller = self.node.get_controller_for_inparent(self.document, self.model)
        if self.in_parent_controller is not None:
            self.vbox.insertWidget(0, self.in_parent_controller.get_widget())

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.ref):
            self.ref.setEditText(none_to_empty(self.node.ref))
        if self.in_parent_controller is not None: self.in_parent_controller.fill_form()

    def save_data_in_model(self):
        if self.in_parent_controller is not None: self.in_parent_controller.save_data_in_model()


class GNCopyChildController(GNodeController):
    def construct_form(self):
        super().construct_form()
        self.construct_group('Operation Settings')
        self.object = self.construct_names_before_self_combo_box('Object:', node_property_name='object')

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.object):
            self.object.setEditText(none_to_empty(self.node.object))


class GNCDeleteController(GNCopyChildController):
    def construct_form(self):
        super().construct_form()
        self.object.setToolTip('&lt;delete <b>object</b>=""/&gt;<br/>' 'Name of the object to delete. Required.')


class GNCReplaceController(GNCopyChildController):
    def construct_form(self):
        super().construct_form()
        self.object.setToolTip('&lt;replace <b>object</b>="" with=""/&gt;<br/>' 'Name of the object to delete (replace). Required.')
        self.replacer = self.construct_names_before_self_combo_box(
            'With:', node_property_name='replacer', display_property_name='name of the object to replace with'
        )
        self.replacer.setToolTip(
            '&lt;replace object="" <b>with</b>=""/&gt;<br/>'
            'Name of the object to replace with. This object does not need to be located in the subtree of the copied object.'
        )

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.replacer):
            self.replacer.setEditText(none_to_empty(self.node.replacer))


class GNCToBlockController(GNMaterialControllerMixin, GNCopyChildController):
    def construct_form(self):
        super().construct_form()
        self.object.setToolTip(
            '&lt;toblock <b>object</b>="" material=""/&gt;<br/>'
            'Name of the object to replace with the the solid block. Required.'
        )
        self.get_material_row()
        material_box, _ = self._construct_hbox('Block material:')
        material_box.addWidget(self.material_selection_type)
        material_box.addWidget(self.material_group)
        self.name = self.construct_line_edit('Name:', node_property_name='name')
        self.name.setToolTip(
            '&lt;{} <b>name</b>="" ...&gt;<br/>'
            'Replacing block name for further reference.'
            ' In the script section, the object is available by GEO table,'
            ' which is indexed by names of geometry objects.'.format(self.node.tag_name(False))
        )
        self.role = self.construct_line_edit('Roles:', node_property_name='role', display_property_name='roles')
        self.role.setToolTip(
            '&lt;{} <b>role</b>="" ...&gt;<br/>'
            'Replacing block role. Important for some solvers.'.format(self.node.tag_name(False))
        )

    def fill_form(self):
        super().fill_form()
        self.name.setText(none_to_empty(self.node.name))
        self.role.setText(none_to_empty(self.node.role))


class GNCSimplifyGradientsController(GNodeController):
    def construct_form(self):
        super().construct_form()
        self.lam = self.construct_line_edit('Wavelength (nm):', node_property_name='lam')
        self.lam.setToolTip(
            '&lt;simplify-gradients <b>lam</b>=""/&gt;<br/>'
            'Wavelength for which simplified gradient is determined. Required.'
        )
        self.linear = self.construct_combo_box('Linear parameter:', ('nr', 'eps'), node_property_name='linear')
        self.linear.setToolTip(
            '&lt;simplify-gradients lam="" <b>linear</b>=""/&gt;<br/>'
            'Specifies which parameter is linear: refractive index (<i>nr</i>) or permittivity (<i>eps</i>).'
        )
        self.linear.lineEdit().setPlaceholderText('nr')
        self.temp = self.construct_line_edit('Temperature (K):', node_property_name='temp')
        self.temp.setToolTip(
            '&lt;simplify-gradients lam="" <b>temp</b>=""/&gt;<br/>'
            'Temperature at which the refractive indices are retrieved.'
        )
        self.temp.setPlaceholderText('300')
        self.dtemp = self.construct_line_edit('Temperature step (K):', node_property_name='dtemp')
        self.dtemp.setToolTip(
            '&lt;simplify-gradients lam="" <b>dtemp</b>=""/&gt;<br/>'
            'Temperature step for approximating temperature dependence of the simplified refractive indices.'
        )
        self.dtemp.setPlaceholderText('100')
        self.only_role = self.construct_line_edit('Only for role:', node_property_name='only_role')
        self.only_role.setToolTip(
            '&lt;simplify-gradients lam="" <b>only-role</b>=""/&gt;<br/>'
            'Only gradients with this role are simplified. If empty, all gradients are simplified.'
        )

    def fill_form(self):
        super().fill_form()
        self.lam.setText(none_to_empty(self.node.lam))
        with BlockQtSignals(self.linear):
            self.linear.setEditText(none_to_empty(self.node.linear))
        self.temp.setText(none_to_empty(self.node.temp))
        self.only_role.setText(none_to_empty(self.node.only_role))


class GNCopyController(GNObjectController):

    have_mesh_settings = False

    def construct_form(self):
        self.construct_group('Copy Settings')
        self.source = self.construct_names_before_self_combo_box('From:', node_property_name='source')
        self.source.setToolTip(
            '&lt;copy <b>from</b>="" ...&gt;<br/>'
            'Name of the source two or three dimensional object to make modified copy of.'
            ' Usually it is some container that has some other named its items or sub-items.'
            ' Required.'
        )
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.source):
            self.source.setEditText(none_to_empty(self.node.source))
