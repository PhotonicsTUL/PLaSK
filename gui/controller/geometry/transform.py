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

from .object import GNObjectController
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty


class GNClipController(GNObjectController):

    def construct_form(self):
        self.construct_group('Clipping Box')
        sign = '-'
        for b in self.node.bound_names():
            setattr(self, b, self.construct_line_edit(b.title()+':', unit=u'µm', node_property_name=b))
            getattr(self, b).setToolTip(u'&lt;clip <b>{0}</b>=""...&gt;<br/>'
                    u'{0} edge of the clipping rectangle. (float (µm), {1}INF by default)'.format(b, sign))
            sign = '+' if sign == '-' else '-'
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        for b in self.node.bound_names():
            getattr(self, b).setText(none_to_empty(getattr(self.node, b)))


class GNFlipMirrorController(GNObjectController):

    def _save_axis_undoable(self):
        self._set_node_by_setter_undoable(lambda n, v: n.set_axis(v),
                                          empty_to_none(
                                              self.axis.currentText()), self.node.axis, 'change axis property')
        #self.node.set_axis(empty_to_none(self.axis.currentText()))

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.axis):
            self.axis.setEditText(none_to_empty(self.node.axis_str()))


class GNFlipController(GNFlipMirrorController):

    def construct_form(self):
        self.construct_group('Flip Settings')
        self.axis = self.construct_combo_box('Flipped axis', items=self.node.get_axes_conf_dim(), change_cb=self._save_axis_undoable)
        self.axis.setToolTip('&lt;flip <b>axis</b>="" ...&gt;<br/>'
                    'Name of the inverted axis (i.e. perpendicular to the reflection plane). (required)')
        super().construct_form()


class GNMirrorController(GNFlipMirrorController):

    def construct_form(self):
        self.construct_group('Mirror Settings')
        self.axis = self.construct_combo_box('Mirrored axis', items=self.node.get_axes_conf_dim(), change_cb=self._save_axis_undoable)
        self.axis.setToolTip('&lt;mirror <b>axis</b>="" ...&gt;<br/>'
                             'Name of the mirrored axis (i.e. perpendicular to the reflection plane). (required)')
        super().construct_form()


class GNExtrusionController(GNObjectController):

    def construct_form(self):
        self.construct_group('Extrusion Settings')
        self.length = self.construct_line_edit('Length:', unit=u'µm', node_property_name='length', display_property_name='extrusion length')
        self.length.setToolTip(u'&lt;extrusion <b>length</b>="" ...&gt;<br/>'
                               u'Length of the extrusion. (float (µm), required)')
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        self.length.setText(none_to_empty(self.node.length))


class GNRevolutionController(GNObjectController):

    def construct_form(self):
        self.construct_group('Revolution Settings')
        self.auto_clip = self.construct_combo_box('Auto clip:', items=['', 'yes', 'no'],
                                                  node_property_name='auto_clip', display_property_name='auto clip')
        self.auto_clip.setToolTip(u'&lt;revolution <b>auto-clip</b>="" ...&gt;<br/>'
                                  u'The value of this attribute can be either \'yes\' of \'no\'.'
                                  u' It specifies whether the item will be implicitly clipped to non-negative '
                                  u'transverse coordinates. Defaults to \'no\'.')
        super().construct_form()
        self.rev_step_num = self.construct_line_edit('Maximum steps number:', node_property_name='rev_step_num',
                                                     display_property_name='maximum steps number')
        self.rev_step_num.setToolTip(u'&lt;revolution <b>rev-steps-num</b>="" rev-steps-dist="" ...&gt;<br/>'
                                     u'Maximum number of the mesh steps in horizontal directions the revolution is '
                                     u'divided into. (integer)')
        self.rev_step_num.setPlaceholderText('10')
        self.rev_step_dist = self.construct_line_edit('Minimum step size:', node_property_name='rev_step_dist',
                                                      display_property_name='minimum step size', unit=u'µm')
        self.rev_step_dist.setToolTip(u'&lt;revolution rev-steps-num="" <b>rev-steps-dist</b>="" ...&gt;<br/>'
                                      u'Minimum step size in horizontal directions.')
        self.rev_step_dist.setPlaceholderText('0.005')

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.auto_clip, self.rev_step_num, self.rev_step_dist):
            self.auto_clip.setEditText(none_to_empty(self.node.auto_clip))
            self.rev_step_num.setText(none_to_empty(self.node.rev_step_num))
            self.rev_step_dist.setText(none_to_empty(self.node.rev_step_dist))


class GNTranslationController(GNObjectController):

    def construct_form(self):
        self.construct_group('Translation Settings')
        def setter(n, v): n.vector = v
        weakself = weakref.proxy(self)
        self.vector = self.construct_point_controllers(row_name='Vector', change_cb=lambda point, _:
            weakself._set_node_by_setter_undoable(setter, list(point),
                                                  weakself.node.vector, 'change translation vector'))
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        for i in range(0, self.node.dim):
            self.vector[i].setText(none_to_empty(self.node.vector[i]))


class GNArrangeController(GNObjectController):

    def construct_form(self):
        self.construct_group('Arrange Settings')
        def setter(n, v): n.step = v
        weakself = weakref.proxy(self)
        self.step = self.construct_point_controllers(row_name='Step', change_cb=lambda point, _:
            weakself._set_node_by_setter_undoable(setter, list(point), weakself.node.step, 'change step in arrange'))
        self.count = self.construct_line_edit('Count:', node_property_name='count')
        self.count.setToolTip(u'&lt;arrange <b>count</b>="" ...&gt;<br/>'
                               u'Number of item repetitions.')
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        for i in range(0, self.node.dim):
            self.step[i].setText(none_to_empty(self.node.step[i]))
        self.count.setText(none_to_empty(self.node.count))
