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

from ...qt import QtGui

from .object import GNObjectController
from .node import GNodeController, GNChildController
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty
from ...model.geometry.reader import GNAligner


class GNGeometryController(GNObjectController):

    def construct_border_controllers(self, row_name=None):
        hbox, group = self._construct_hbox(row_name)
        res = tuple(self.construct_material_combo_box(items=['', 'mirror', 'periodic', 'extend'])
                    for _ in range(0, 2))
        for w in res: hbox.addWidget(w)
        if row_name:
            return res
        else:
            return res, group

    def fill_form(self):
        self.construct_group('Border Settings')
        self.borders = tuple(self.construct_border_controllers('{}/{}:'.format(lo.title(), hi.title()))
                             for (lo, hi) in self.node.get_alternative_direction_names())
        super(GNGeometryController, self).fill_form()

    def save_data_in_model(self):
        super(GNGeometryController, self).save_data_in_model()
        for dir in range(0, self.node.dim):
            for lh in range(0, 2):
                self.node.borders[dir][lh] = empty_to_none(self.borders[dir][lh].currentText())

    def on_edit_enter(self):
        super(GNGeometryController, self).on_edit_enter()
        for dir in range(0, self.node.dim):
            for lh in range(0, 2):
                with BlockQtSignals(self.borders[dir][lh]):
                    self.borders[dir][lh].setEditText(none_to_empty(self.node.borders[dir][lh]))


class GNCartesian2DGeometryController(GNGeometryController):

    def fill_form(self):
        self.construct_group('Implicit Extrusion Settings')
        self.length = self.construct_line_edit('Length:', unit=u'µm', node_property_name='length', display_property_name='longitudinal dimension of the geometry')
        self.length.setToolTip(u'&lt;cartesian2d <b>length</b>="" ...&gt;<br/>'
            u'Longitudinal dimension of the geometry (float [µm]). Default value is: +infty.')
        super(GNCartesian2DGeometryController, self).fill_form()

    def save_data_in_model(self):
        super(GNCartesian2DGeometryController, self).save_data_in_model()
        #self.node.length = empty_to_none(self.length.text())

    def on_edit_enter(self):
        super(GNCartesian2DGeometryController, self).on_edit_enter()
        self.length.setText(none_to_empty(self.node.length))