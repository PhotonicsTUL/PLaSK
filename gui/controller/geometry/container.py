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
from .node import GNodeController
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty
from ...model.geometry.reader import GNAligner


class GNGapController(GNodeController):

    def fill_form(self):
        super(GNGapController, self).fill_form()
        self.gap_type = QtGui.QComboBox()
        self.gap_type.addItems(['gap size', 'total container size'])
        self.gap_type.currentIndexChanged.connect(self.after_field_change)
        self.gap_value = self.construct_line_edit()
        self.form_layout.addRow(self.gap_type, self.gap_value)

    def save_data_in_model(self):
        super(GNGapController, self).save_data_in_model()
        self.node.size_is_total = self.gap_type.currentIndex() == 1
        self.node.size = empty_to_none(self.gap_value.text())

    def on_edit_enter(self):
        super(GNGapController, self).on_edit_enter()
        with BlockQtSignals(self.gap_type) as ignored:
            self.gap_type.setCurrentIndex(1 if self.node.size_is_total else 0)
            self.gap_value.setText(none_to_empty(self.node.size))
            

class GNShelfController(GNObjectController):
    
    def fill_form(self):
        self.construct_group('Shelf-specific settings')
        self.repeat = self.construct_line_edit('repeat')
        self.shift = self.construct_line_edit('shift')
        self.flat = self.construct_combo_box('flat', items=['', 'yes', 'no'])
        super(GNShelfController, self).fill_form()

    def save_data_in_model(self):
        super(GNShelfController, self).save_data_in_model()
        self.node.repeat = empty_to_none(self.repeat.text())
        self.node.shift = empty_to_none(self.shift.text())
        self.node.flat = empty_to_none(self.flat.currentText())

    def on_edit_enter(self):
        super(GNShelfController, self).on_edit_enter()
        self.repeat.setText(none_to_empty(self.node.repeat))
        self.shift.setText(none_to_empty(self.node.shift))
        self.flat.setEditText(none_to_empty(self.node.flat))


class GNStackController(GNObjectController):

    def fill_form(self):
        self.construct_group('Stack-specific settings')
        self.repeat = self.construct_line_edit('repeat')
        self.shift = self.construct_line_edit('shift')

        self.pos_layout = self.construct_group('Default children position')
        self.position = []
        axes_conf = self.node.get_axes_conf()
        for c in self.node.aligners_dir():
            position = QtGui.QComboBox()
            position.addItems(
                [('{} origin at' if i == 3 else '{} at').format(x)
                for i, x in enumerate(GNAligner.names(self.node.children_dim, axes_conf, c, False))]
            )
            position.currentIndexChanged.connect(self.after_field_change)
            pos_value = self.construct_line_edit()
            self.pos_layout.addRow(position, pos_value)
            self.position.append((position, pos_value))

        #self.child_pos = self.construct_combo_box('flat', items=['', 'yes', 'no'])
        super(GNStackController, self).fill_form()

    def save_data_in_model(self):
        super(GNStackController, self).save_data_in_model()
        self.node.repeat = empty_to_none(self.repeat.text())
        self.node.shift = empty_to_none(self.shift.text())
        for i, pos in enumerate(self.position):
            val = pos[1].text()
            if val:
                self.node.aligners[i] = GNAligner(pos[0].currentIndex(), val)

    def on_edit_enter(self):
        super(GNStackController, self).on_edit_enter()
        self.repeat.setText(none_to_empty(self.node.repeat))
        self.shift.setText(none_to_empty(self.node.shift))
        for i, pos in enumerate(self.position):
            aligner = self.node.aligners[i]
            if aligner.position is not None:
                pos[0].setCurrentIndex(aligner.position)
            pos[1].setText(none_to_empty(aligner.value))