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
import weakref

from .object import GNObjectController
from ...utils.str import empty_to_none, none_to_empty
from ...qt.QtWidgets import *
from ...utils.qsignals import BlockQtSignals
from ...utils.widgets import ComboBox


class GNMaterialControllerMixin:

    def get_material_row(self):
        self.material_selection_type = ComboBox()
        self.material_selection_type.addItems(['Solid', 'Vertical Gradient'])
        self.material_selection_type.currentIndexChanged.connect(self._material_type_changed)

        self.material_solid = self.construct_material_combo_box(change_cb=self._save_material_in_model_undoable)

        material_tb_hbox, material_tb_group = self._construct_hbox()
        self.material_bottom = self.construct_material_combo_box(change_cb=self._save_material_in_model_undoable)
        self.material_top = self.construct_material_combo_box(change_cb=self._save_material_in_model_undoable)
        self.material_shape = self.construct_line_edit(None, node_property_name='material_shape',
                                                       display_property_name='material shape exponent',
                                                       change_cb=self._save_material_in_model_undoable)
        self.material_shape.setToolTip(u'&lt;{} <b>material-shape</b>="" ...&gt;<br/>'
                                 u'Shape exponent of changing material. Setting this value to anything different than '
                                 u'one allows to specify non-linearly varying material. (float)'
                                 .format(self.node.tag_name(False)))
        self.material_shape.setPlaceholderText('1')
        self.material_bottom.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.material_top.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.material_shape.setMaximumWidth(50)
        material_tb_hbox.addWidget(self.material_bottom)
        material_tb_hbox.addWidget(QLabel('...'))
        material_tb_hbox.addWidget(self.material_shape)
        material_tb_hbox.addWidget(QLabel('...'))
        material_tb_hbox.addWidget(self.material_top)

        self.material_group = QStackedWidget()
        self.material_group.addWidget(self.material_solid)
        self.material_group.addWidget(material_tb_group)
        self.material_selection_type.currentIndexChanged.connect(self.material_group.setCurrentIndex)

        return self.material_selection_type, self.material_group

    def select_info(self, info):
        prop = getattr(info, 'property')
        if prop == 'material':
            self.material_selection_type.setFocus()
        elif prop in ('material_shape', 'material_bottom', 'material_top'):
            self.material_selection_type.setCurrentIndex(1)
            getattr(self, prop).setFocus()
        else:
            super().select_info(info)

    def _material_type_changed(self):
        if self.material_selection_type.currentIndex() == 1:
            with BlockQtSignals(self.material_bottom, self.material_top):
                material = self.material_solid.currentText()
                self.material_bottom.setEditText(material)
                self.material_top.setEditText(material)
        else:
            with BlockQtSignals(self.material_solid):
                self.material_solid.setEditText(self.material_bottom.currentText())
        self._save_material_in_model_undoable()

    def _save_material_in_model_undoable(self):
        def setter(n, v):
            n.material_bottom = v[0]
            n.material_top = v[1]
            n.material_shape = v[2]

        if self.material_selection_type.currentIndex() == 0:
            m = empty_to_none(self.material_solid.currentText())
            new_material = (m, m, None)
        else:
            new_material = (empty_to_none(self.material_bottom.currentText()),
                            empty_to_none(self.material_top.currentText()),
                            empty_to_none(self.material_shape.text()))

        self._set_node_by_setter_undoable(
            setter, new_material, (self.node.material_bottom, self.node.material_top, self.node.material_shape),
            'change material')

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.material_selection_type,
                            self.material_solid, self.material_bottom, self.material_top, self.material_shape):
            index = 0 if self.node.is_solid() else 1
            self.material_selection_type.setCurrentIndex(index)
            self.material_group.setCurrentIndex(index)
            if self.node.is_solid():
                self.material_solid.setEditText(none_to_empty(self.node.material_bottom))
            else:
                self.material_bottom.setEditText(none_to_empty(self.node.material_bottom))
                self.material_top.setEditText(none_to_empty(self.node.material_top))
                self.material_shape.setText(none_to_empty(self.node.material_shape))


class GNLeafController(GNMaterialControllerMixin, GNObjectController):

    def construct_form(self):
        material_form = self.construct_group('Material')
        material_form.addRow(*self.get_material_row())
        super().construct_form()


class GNBlockController(GNLeafController):

    def construct_form(self):
        if self.node.dim == 2:
            name = 'Rectangle'
            field_names = 'width', 'height'
        else:
            name = 'Cuboid'
            field_names = 'depth', 'width', 'height'
        self.construct_group('{} Settings'.format(name))
        def setter(n, v): n.size = v
        weakself = weakref.proxy(self)
        self.size = self.construct_point_controllers(row_name='Size', field_names=field_names, change_cb=lambda point:
            weakself._set_node_by_setter_undoable(setter, list(point), weakself.node.size, 'change block size'),
        )
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        for i in range(0, self.node.dim):
            self.size[i].setText(none_to_empty(self.node.size[i]))


class GNTriangleController(GNLeafController):

    def _on_point_set(self, index, value):
        def setter(n, v): n.points = n.points[0:index] + (v,) + n.points[index+1:]
        self._set_node_by_setter_undoable(setter, value, self.node.points[index],
            'change {} triangle point'.format('first' if index == 0 else 'second')
        )

    def construct_form(self):
        self.construct_group('Vertex Coordinates (other than: {}):'
                             .format(', '.join('0' for _ in range(0, self.node.dim))))
        weakself = weakref.proxy(self)
        self.points = (self.construct_point_controllers(row_name='First',
                                                        change_cb=lambda point: weakself._on_point_set(0, point)),
                      self.construct_point_controllers(row_name='Second',
                                                       change_cb=lambda point: weakself._on_point_set(1, point)))
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        for i in range(0, self.node.dim):
            self.points[0][i].setText(none_to_empty(self.node.points[0][i]))
            self.points[1][i].setText(none_to_empty(self.node.points[1][i]))


class GNCircleController(GNLeafController):

    def construct_form(self):
        self.construct_group('{} Size:'.format('Circle' if self.node.dim == 2 else 'Sphere'))
        self.radius = self.construct_line_edit('Radius:', unit=u'µm', node_property_name='radius')
        self.radius.setToolTip(u'&lt;{} <b>radius</b>="" ...&gt;<br/>'
                               u'Radius. (float [µm], required)'.format(self.node.tag_name(False)))
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        self.radius.setText(none_to_empty(self.node.radius))


class GNCylinderController(GNLeafController):

    def construct_form(self):
        self.construct_group('Cylinder Size')
        self.radius = self.construct_line_edit('Radius:', unit=u'µm', node_property_name='radius',
                                               display_property_name='radius of the cylinder base')
        self.radius.setToolTip(u'&lt;cylinder <b>radius</b>="" height="" ...&gt;<br/>'
                               u'Radius of the cylinder base. (float [µm], required)')
        self.height = self.construct_line_edit('Height:', unit=u'µm', node_property_name='height',
                                               display_property_name='height of the cylinder')
        self.height.setToolTip(u'&lt;cylinder radius="" <b>height</b>="" ...&gt;<br/>'
                               u'Height of the cylinder. (float [µm], required)')
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        self.radius.setText(none_to_empty(self.node.radius))
        self.height.setText(none_to_empty(self.node.height))


class GNPrismController(GNLeafController):

    def _on_point_set(self, index, value):
        def setter(n, v): n.points = n.points[0:index] + (v,) + n.points[index+1:]
        self._set_node_by_setter_undoable(setter, value, self.node.points[index],
            'change {} triangle point'.format('first' if index == 0 else 'second')
        )

    def construct_form(self):
        self.construct_group('Base Vertex Coordinates (other than: {}):'
                             .format(', '.join('0' for _ in range(0, self.node.dim))))
        weakself = weakref.proxy(self)
        self.points = (self.construct_point_controllers(row_name='First', field_names=('longitudinal', 'transverse'),
                                                        change_cb=lambda point: weakself._on_point_set(0, point)),
                       self.construct_point_controllers(row_name='Second', field_names=('longitudinal', 'transverse'),
                                                        change_cb=lambda point: weakself._on_point_set(1, point)))
        self.construct_group('Prism Settings')
        self.height = self.construct_line_edit('Prism Height:', unit=u'µm', node_property_name='height',
                                               display_property_name='height of the prism')
        self.height.setToolTip(u'&lt;prism radius="" <b>height</b>="" ...&gt;<br/>'
                               u'Height of the prism. (float [µm], required)')

        super().construct_form()

    def fill_form(self):
        super().fill_form()
        for i in range(0, self.node.dim):
            self.points[0][i].setText(none_to_empty(self.node.points[0][i]))
            self.points[1][i].setText(none_to_empty(self.node.points[1][i]))
        self.height.setText(none_to_empty(self.node.height))
