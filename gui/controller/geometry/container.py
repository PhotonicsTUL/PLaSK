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

import weakref

from ...qt.QtWidgets import *

from .object import GNObjectController
from .node import GNodeController, GNChildController, aligners_to_controllers
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty
from ...utils.widgets import ComboBox


class GNGapController(GNodeController):

    @staticmethod
    def _set_gap_params(node, params):
        node.size_is_total = params[0]
        node.size = params[1]

    def _on_change_gap_params(self):
        new_value = (self.gap_type.currentIndex() == 1, empty_to_none(self.gap_value.text()))
        old_value = (self.node.size_is_total, self.node.size)
        self._set_node_by_setter_undoable(
            GNGapController._set_gap_params, new_value, old_value,
            u'change gap size to: {}="{}"µm'.format('total' if new_value[0] else 'gap', none_to_empty(new_value[1]))
        )

    def construct_form(self):
        super().construct_form()
        self.gap_type = ComboBox()
        self.gap_type.addItems(['Gap size', 'Total container size'])
        self.gap_type.currentIndexChanged.connect(self._on_change_gap_params)
        self.gap_value = self.construct_line_edit(self.gap_type, unit=u'µm', change_cb=self._on_change_gap_params)

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.gap_type):
            self.gap_type.setCurrentIndex(1 if self.node.size_is_total else 0)
            self.gap_value.setText(none_to_empty(self.node.size))

    def select_info(self, info):
        prop = getattr(info, 'property')
        if prop == 'size':
            self.gap_value.setFocus()
        else:
            super().select_info(info)


class GNShelfController(GNObjectController):

    def construct_form(self):
        self.construct_group('Shelf Settings')
        self.repeat = self.construct_line_edit(
            'Repeat:', node_property_name='repeat', display_property_name='number of repetitive occurrences'
        )
        self.repeat.setToolTip(
            '&lt;shelf <b>repeat</b>="" ...&gt;<br/>'
            'Number of repetitive occurrences of stack content.'
            ' This attribute allows to create periodic horizontal structures easily.'
            ' Defaults to 1. (integer)'
        )
        self.shift = self.construct_line_edit('Shift:', unit=u'µm', node_property_name='shift')
        self.shift.setToolTip(
            u'&lt;shelf <b>shift</b>="" ...&gt;<br/>'
            u'Horizontal position of the shelf left edge in its local coordinates.'
            u' Defaults to 0. (float, µm)'
        )
        self.flat = self.construct_combo_box('Flat:', items=['', 'yes', 'no'], node_property_name='flat')
        self.flat.setToolTip(
            u'&lt;shelf <b>flat</b>="" ...&gt;<br/>'
            u'The value of this attribute can be either true of false.'
            u' It specifies whether all the items in the shelf are required to have the same height'
            u' (therefore the top edge of the shelf is flat). Defaults to true.'
        )
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        self.repeat.setText(none_to_empty(self.node.repeat))
        self.shift.setText(none_to_empty(self.node.shift))
        with BlockQtSignals(self.flat):
            self.flat.setEditText(none_to_empty(self.node.flat))


class GNAlignContainerController(GNObjectController):

    def construct_form(self):
        self.pos_layout = self.construct_group("Items' Placement")
        self.order = self.construct_combo_box('Ordering:', items=['', 'normal', 'reverse'], node_property_name='order')
        self.order.setToolTip(
            u'&lt;align <b>order</b>="" ...&gt;<br/>'
            u'Order of items in the container. If <tt>normal</tt>, the items lower in the list override the ones previous ones.'
            u' <tt>reverse</tt> means that each item is on top of all the later ones.'
            u' Defaults to ``normal``.'
        )
        weakself = weakref.proxy(self)
        def setter(n, v):
            n.aligners = v
        self.positions = self.construct_align_controllers(
            change_cb=lambda aligners: weakself.
            _set_node_by_setter_undoable(setter, aligners, weakself.node.aligners, 'change default items positions')
        )
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.order):
            self.order.setEditText(none_to_empty(self.node.order))
        aligners_to_controllers(self.node.aligners, self.positions)


class GNStackController(GNObjectController):

    def construct_form(self):
        self.construct_group('Stack Settings')
        self.repeat = self.construct_line_edit(
            'Repeat:', node_property_name='repeat', display_property_name='number of repetitive occurrences'
        )
        self.repeat.setToolTip(
            '&lt;stack <b>repeat</b>="" ...&gt;<br/>'
            'Number of repetitive occurrences of stack content.'
            ' This attribute allows to create periodic vertical structures (e. g. DBRs) easily.'
            ' Defaults to 1. (integer))'
        )
        self.shift = self.construct_line_edit('Shift:', node_property_name='shift')
        self.shift.setToolTip(
            u'&lt;stack <b>shift</b>="" ...&gt;<br/>'
            u'Vertical position of the stack bottom edge in its local coordinates.'
            u' Defaults to 0. (float, µm)'
        )
        self.pos_layout = self.construct_group("Default Items' Positions")

        def setter(n, v):
            n.aligners = v

        weakself = weakref.proxy(self)
        self.positions = self.construct_align_controllers(
            change_cb=lambda aligners: weakself.
            _set_node_by_setter_undoable(setter, aligners, self.node.aligners, 'change default items positions in stack')
        )
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        self.repeat.setText(none_to_empty(self.node.repeat))
        self.shift.setText(none_to_empty(self.node.shift))
        aligners_to_controllers(self.node.aligners, self.positions)


class GNContainerChildController(GNChildController):

    def construct_form_other(self):
        pass

    def construct_form(self):
        self.construct_group('Placement in Parent Container')

        def setter(n, v):
            n.in_parent_aligners = v

        weakself = weakref.proxy(self)
        self.positions = self.construct_align_controllers(
            change_cb=lambda aligners: weakself._set_node_by_setter_undoable(
                setter, aligners, weakself.child_node.in_parent_aligners, 'change item position', node=self.child_node
            )
        )
        self.construct_form_other()
        self.path = self.construct_combo_box(
            'Path:',
            items=[''] + sorted(self.model.get_paths(), key=lambda s: s.lower()),
            node_property_name='path',
            node=self.child_node
        )
        self.path.setToolTip(
            'Name of a path that can be later on used to distinguish '
            'between multiple occurrences of the same object.'
        )

    def fill_form(self):
        super().fill_form()
        aligners_to_controllers(self.child_node.in_parent_aligners, self.positions)
        with BlockQtSignals(self.path):
            self.path.setEditText(none_to_empty(self.child_node.path))


class GNStackChildController(GNContainerChildController):

    def _set_node_zero_undoable(self):
        new_value = empty_to_none(self.zero.text())
        old_value = self.child_node.in_parent_attrs.get('zero')
        action_name = "change {}'s zero to local {} in {}".format(
            self.node.display_name(full_name=False).lower(), self.zero.text(),
            self.child_node.display_name(full_name=False).lower()
        )
        self._set_node_by_setter_undoable(
            lambda n, v: n.in_parent_attrs.__setitem__('zero', v),
            new_value,
            old_value,
            action_name=action_name,
            node=self.child_node
        )

    def construct_form_other(self):
        weakself = weakref.proxy(self)
        name = self.node.display_name(full_name=False)
        direction = {'Shelf': 'horizontal'}.get(name, 'vertical')
        self.zero = self.construct_line_edit(
            "{}'s zero align:".format(name),
            change_cb=lambda: weakself._set_node_zero_undoable(),
            unit="µm",
            use_defines_completer=True
        )
        self.zero.setToolTip(
            "Shift {0} {1}ly so its zero aligns with the specified {1} position in this item. "
            "Only one object in the {0} can have this value not empty and you must not "
            "use any other method of defining the {0} {1} coordinates.".format(name.lower(), direction)
        )

    def fill_form(self):
        super().fill_form()
        self.zero.setText(none_to_empty(self.child_node.in_parent_attrs.get('zero')))
