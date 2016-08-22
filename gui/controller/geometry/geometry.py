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
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty

class GNGeometryController(GNObjectController):

    def _borders_to_model_undoable(self):
        self._set_node_property_undoable('edges',
            [[empty_to_none(self.edges[dir][lh].currentText()) for lh in range(0, 2)] for dir in range(0, self.node.dim)],
            action_name='change geometry edges')

    def construct_border_controllers(self, row_name=None):
        hbox, group = self._construct_hbox(row_name)
        res = tuple(self.construct_material_combo_box(items=['', 'mirror', 'periodic', 'extend'], change_cb=self._borders_to_model_undoable)
                    for _ in range(0, 2))
        for r in res:
            r.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        for w in res: hbox.addWidget(w)
        return res if row_name else (res, group)

    def construct_form(self):
        self.construct_group('Edges Settings')
        self.edges = tuple(self.construct_border_controllers('{}/{}:'.format(lo.title(), hi.title()))
                             for (lo, hi) in self.node.get_alternative_direction_names())
        super(GNGeometryController, self).construct_form(roles=False)

    def fill_form(self):
        super(GNGeometryController, self).fill_form()
        for dir in range(0, self.node.dim):
            for lh in range(0, 2):
                with BlockQtSignals(self.edges[dir][lh]):
                    self.edges[dir][lh].setEditText(none_to_empty(self.node.edges[dir][lh]))


class GNCartesian2DGeometryController(GNGeometryController):

    def construct_form(self):
        self.construct_group('Implicit Extrusion Settings')
        self.length = self.construct_line_edit('Length:', unit=u'µm', node_property_name='length', display_property_name='longitudinal dimension of the geometry')
        self.length.setToolTip(u'&lt;cartesian2d <b>length</b>="" ...&gt;<br/>'
            u'Longitudinal dimension of the geometry (float [µm]). Default value is: +infty.')
        super(GNCartesian2DGeometryController, self).construct_form()

    def fill_form(self):
        super(GNCartesian2DGeometryController, self).fill_form()
        self.length.setText(none_to_empty(self.node.length))
