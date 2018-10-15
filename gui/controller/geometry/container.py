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

from ...qt.QtWidgets import *

from .object import GNObjectController
from .node import GNodeController, GNChildController, aligners_to_controllers
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty


class GNGapController(GNodeController):

    @staticmethod
    def _set_gap_params(node, params):
        node.size_is_total = params[0]
        node.size = params[1]

    def _on_change_gap_params(self):
        new_value = (self.gap_type.currentIndex() == 1, empty_to_none(self.gap_value.text()))
        old_value = (self.node.size_is_total, self.node.size)
        self._set_node_by_setter_undoable(GNGapController._set_gap_params, new_value, old_value,
            u'change gap size to: {}="{}"µm'.format('total' if new_value[0] else 'gap', none_to_empty(new_value[1])))

    def construct_form(self):
        super(GNGapController, self).construct_form()
        self.gap_type = QComboBox()
        self.gap_type.addItems(['Gap size', 'Total container size'])
        self.gap_type.currentIndexChanged.connect(self._on_change_gap_params)
        self.gap_value = self.construct_line_edit(self.gap_type, unit=u'µm', change_cb=self._on_change_gap_params)

    def fill_form(self):
        super(GNGapController, self).fill_form()
        with BlockQtSignals(self.gap_type):
            self.gap_type.setCurrentIndex(1 if self.node.size_is_total else 0)
            self.gap_value.setText(none_to_empty(self.node.size))

    def select_info(self, info):
        prop = getattr(info, 'property')
        if prop == 'size':
            self.gap_value.setFocus()
        else:
            super(GNGapController, self).select_info(info)
            

class GNShelfController(GNObjectController):
    
    def construct_form(self):
        self.construct_group('Shelf Settings')
        self.repeat = self.construct_line_edit('Repeat:', node_property_name='repeat',
                                               display_property_name='number of repetitive occurrences')
        self.repeat.setToolTip('&lt;shelf <b>repeat</b>="" ...&gt;<br/>'
                               'Number of repetitive occurrences of stack content.'
                               ' This attribute allows to create periodic horizontal structures easily.'
                               ' Defaults to 1. (integer)')
        self.shift = self.construct_line_edit('Shift:', unit=u'µm', node_property_name='shift')
        self.shift.setToolTip(u'&lt;shelf <b>shift</b>="" ...&gt;<br/>'
                              u'Horizontal position of the shelf left edge in its local coordinates.'
                              u' Defaults to 0. (float [µm])')
        self.flat = self.construct_combo_box('Flat:', items=['', 'yes', 'no'], node_property_name='flat')
        self.flat.setToolTip(u'&lt;shelf <b>flat</b>="" ...&gt;<br/>'
                             u'The value of this attribute can be either true of false.'
                             u' It specifies whether all the items in the shelf are required to have the same height'
                             u' (therefore the top edge of the shelf is flat). Defaults to true.')
        super(GNShelfController, self).construct_form()

    def fill_form(self):
        super(GNShelfController, self).fill_form()
        self.repeat.setText(none_to_empty(self.node.repeat))
        self.shift.setText(none_to_empty(self.node.shift))
        with BlockQtSignals(self.flat):
            self.flat.setEditText(none_to_empty(self.node.flat))


class GNContainerController(GNObjectController):

    def construct_form(self):
        self.pos_layout = self.construct_group('Default Items Positions')
        weakself = weakref.proxy(self)
        def setter(n, v): n.aligners = v
        self.positions = self.construct_align_controllers(change_cb=lambda aligners:
            weakself._set_node_by_setter_undoable(setter, aligners,
                                                  weakself.node.aligners, 'change default items positions')
        )
        super(GNContainerController, self).construct_form()

    def fill_form(self):
        super(GNContainerController, self).fill_form()
        aligners_to_controllers(self.node.aligners, self.positions)


class GNStackController(GNObjectController):

    def construct_form(self):
        self.construct_group('Stack Settings')
        self.repeat = self.construct_line_edit('Repeat:', node_property_name='repeat',
                                               display_property_name='number of repetitive occurrences')
        self.repeat.setToolTip('&lt;stack <b>repeat</b>="" ...&gt;<br/>'
                                'Number of repetitive occurrences of stack content.'
                                ' This attribute allows to create periodic vertical structures (e. g. DBRs) easily.'
                                ' Defaults to 1. (integer))')
        self.shift = self.construct_line_edit('Shift:', node_property_name='shift')
        self.shift.setToolTip(u'&lt;stack <b>shift</b>="" ...&gt;<br/>'
                                u'Vertical position of the stack bottom edge in its local coordinates.'
                                u' Defaults to 0. (float [µm])')
        self.pos_layout = self.construct_group('Default Items Positions')
        def setter(n, v): n.aligners = v
        weakself = weakref.proxy(self)
        self.positions = self.construct_align_controllers(change_cb=lambda aligners:
            weakself._set_node_by_setter_undoable(setter, aligners, self.node.aligners,
                                                  'change default items positions in stack')
        )
        super(GNStackController, self).construct_form()

    def fill_form(self):
        super(GNStackController, self).fill_form()
        self.repeat.setText(none_to_empty(self.node.repeat))
        self.shift.setText(none_to_empty(self.node.shift))
        aligners_to_controllers(self.node.aligners, self.positions)


class GNContainerChildBaseController(GNChildController):

    def construct_form(self):
        self.construct_group('Position in Container')
        def setter(n, v): n.in_parent = v
        weakself = weakref.proxy(self)
        self.positions = self.construct_align_controllers(change_cb=lambda aligners:
            weakself._set_node_by_setter_undoable(
                setter, aligners, weakself.child_node.in_parent, 'change item position', node=self.child_node))
        self.path = self.construct_combo_box(
            'Path:', items=[''] + sorted(self.model.get_paths(), key=lambda s: s.lower()),
            node_property_name='path', node=self.child_node)
        self.path.setToolTip('Name of a path that can be later on used to distinguish '
                             'between multiple occurrences of the same object.')

    def fill_form(self):
        super(GNContainerChildBaseController, self).fill_form()
        aligners_to_controllers(self.child_node.in_parent, self.positions)
        with BlockQtSignals(self.path):
            self.path.setEditText(none_to_empty(self.child_node.path))