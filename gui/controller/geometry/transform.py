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
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty


class GNClipController(GNObjectController):

    def fill_form(self):
        self.construct_group('Clipping box')
        sign = '-'
        for b in self.node.bound_names():
            setattr(self, b, self.construct_line_edit(b))
            getattr(self, b).setToolTip(u'&lt;clip <b>{0}</b>=""...&gt;<br/>'
                    u'{0} edge of the clipping rectangle. (float [µm], {1}INF by default)'.format(b, sign))
            sign = '+' if sign == '-' else '-'
        super(GNClipController, self).fill_form()

    def save_data_in_model(self):
        super(GNClipController, self).save_data_in_model()
        for b in self.node.bound_names():
            setattr(self.node, b, empty_to_none(getattr(self, b).text()))

    def on_edit_enter(self):
        super(GNClipController, self).on_edit_enter()
        for b in self.node.bound_names():
            getattr(self, b).setText(none_to_empty(getattr(self.node, b)))


class GNFlipMirrorController(GNObjectController):

    def save_data_in_model(self):
        super(GNFlipMirrorController, self).save_data_in_model()
        self.node.set_axis(empty_to_none(self.axis.currentText()))

    def on_edit_enter(self):
        super(GNFlipMirrorController, self).on_edit_enter()
        with BlockQtSignals(self.axis) as ignored:
            self.axis.setEditText(none_to_empty(self.node.axis_str()))


class GNFlipController(GNFlipMirrorController):

    def fill_form(self):
        self.construct_group('Flip-specific settings')
        self.axis = self.construct_combo_box('inverted axis', items=self.node.get_axes_conf_dim())
        self.axis.setToolTip('&lt;flip <b>axis</b>="" ...&gt;<br/>'
                    'Name of the inverted axis (i.e. perpendicular to the reflection plane). (required)')
        super(GNFlipController, self).fill_form()


class GNMirrorController(GNFlipMirrorController):

    def fill_form(self):
        self.construct_group('Mirror-specific settings')
        self.axis = self.construct_combo_box('inverted axis', items=self.node.get_axes_conf_dim())
        self.axis.setToolTip('&lt;mirror <b>axis</b>="" ...&gt;<br/>'
            'Name of the inverted axis (i.e. perpendicular to the reflection plane). (required)')
        super(GNMirrorController, self).fill_form()


class GNExtrusionController(GNObjectController):

    def fill_form(self):
        self.construct_group('Extrusion-specific settings')
        self.length = self.construct_line_edit('length')
        self.length.setToolTip(u'&lt;extrusion <b>length</b>="" ...&gt;<br/>'
                            u'Length of the extrusion. (float [µm], required)')
        super(GNExtrusionController, self).fill_form()

    def save_data_in_model(self):
        super(GNExtrusionController, self).save_data_in_model()
        self.node.length = empty_to_none(self.length.text())

    def on_edit_enter(self):
        super(GNExtrusionController, self).on_edit_enter()
        self.length.setText(none_to_empty(self.node.length))


class GNRevolutionController(GNObjectController):

    def fill_form(self):
        self.construct_group('Revolution-specific settings')
        self.auto_clip = self.construct_combo_box('Auto clip:', items=['', 'yes', 'no'])
        self.auto_clip.setToolTip(u'&lt;revolution <b>auto_clip</b>="" ...&gt;<br/>'
                                u'The value of this attribute can be either true of false.'
                                u' It specifies whether item will be implicitly clipped to non-negative tran. coordinates'
                                u' Defaults to false.')
        super(GNRevolutionController, self).fill_form()

    def save_data_in_model(self):
        super(GNRevolutionController, self).save_data_in_model()
        self.node.auto_clip = empty_to_none(self.auto_clip.currentText())

    def on_edit_enter(self):
        super(GNRevolutionController, self).on_edit_enter()
        with BlockQtSignals(self.auto_clip):
            self.auto_clip.setEditText(none_to_empty(self.node.auto_clip))