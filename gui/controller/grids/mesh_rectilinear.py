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

from . import GridController
from ...qt.QtWidgets import *
from ...utils import getattr_by_path
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty
from ...model.grids.mesh_rectangular import AXIS_NAMES
from ..defines import get_defines_completer
from ...utils.widgets import ComboBox, TextEdit


class AxisEdit(QGroupBox):

    def _type_changed(self):
        self.controller.fire_changed()

    def __init__(self, controller, axis_path, axis, title=None, allow_type_select=False, accept_non_regular=False):
        super(AxisEdit, self).__init__(title if title is not None else axis)
        if axis is None: axis = 'axis'
        self.controller = controller
        self.axis_path = axis_path
        defines = get_defines_completer(controller.document.defines.model, self)
        form_layout = QFormLayout()
        self.allow_type_select = allow_type_select
        if not allow_type_select:
            self.accept_non_regular = accept_non_regular
        if self.allow_type_select:
            self.type = ComboBox()
            self.type.addItems(['(auto-detected)', 'ordered', 'regular'])
            self.type.setEditable(True)
            self.type.setToolTip('&lt;{} <b>type</b>=""&gt;<br/>'
                                 'Type of axis. If auto-detected is selected, axis will be regular only if any of the '
                                 'start, stop or num attributes are specified (in other case it will be ordered).'
                                 .format(axis))
            self.type.editingFinished.connect(
                lambda : self.controller._change_attr((axis_path, 'type'),
                                                      None if self.type.currentIndex() == 0 else empty_to_none(self.type.currentText())))
            self.type.currentIndexChanged.connect(self._auto_enable_points)
            form_layout.addRow("Axis type:", self.type)
        range_layout = QHBoxLayout()
        self.start = QLineEdit()
        self.start.setCompleter(defines)
        self.start.setToolTip(u'&lt;{} <b>start</b>="" stop="" num=""&gt;<br/>'
                              u'Position of the first point on the axis. (float [µm])'.format(axis))
        self.start.editingFinished.connect(lambda : self.controller._change_attr((axis_path, 'start'), empty_to_none(self.start.text())))
        # range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.start)
        self.stop = QLineEdit()
        self.stop.setCompleter(defines)
        self.stop.setToolTip(u'&lt;{} start="" <b>stop</b>="" num=""&gt;\n'
                             u'Position of the last point on the axis. (float [µm])'.format(axis))
        self.stop.editingFinished.connect(lambda : self.controller._change_attr((axis_path, 'stop'), empty_to_none(self.stop.text())))
        range_layout.addWidget(QLabel(" Stop:"))
        range_layout.addWidget(self.stop)
        self.num = QLineEdit()
        self.num.setCompleter(defines)
        self.num.setToolTip('&lt;{} start="" stop="" <b>num</b>=""&gt;<br/>'
                            'Number of the equally distributed points along the axis. (integer)'.format(axis))
        self.num.editingFinished.connect(lambda : self.controller._change_attr((axis_path, 'num'), empty_to_none(self.num.text())))
        range_layout.addWidget(QLabel(" Num:"))
        range_layout.addWidget(self.num)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_widget = QWidget()
        range_widget.setLayout(range_layout)
        form_layout.addRow("Start:", range_widget)
        if allow_type_select or accept_non_regular:
            self.points = TextEdit()
            self.points.setToolTip('&lt;{0}&gt;<b><i>points</i></b>&lt;/{0}&gt;<br/>'
                                   'Comma-separated list of the mesh points along this axis.'.format(axis))
            # self.points.editingFinished.connect(controller.fire_changed)
            #self.points.setWordWrapMode(QTextEdit.LineWrapMode)
            form_layout.addRow("Points:", self.points)
            if allow_type_select:
                self.points.setVisible(self.are_points_editable())
            #self.points
            self.points.editingFinished.connect(
                lambda : self.controller._change_attr((axis_path, 'points'), empty_to_none(self.points.toPlainText())))
            self._auto_enable_points()
        self.setLayout(form_layout)

    def are_points_editable(self):
        if self.allow_type_select:
            return self.type.currentText() != 'regular'
        else:
            return self.accept_non_regular

    # def to_model(self, axis_model):
    #     if self.allow_type_select:
    #         if self.type.currentIndex() == 0:
    #             axis_model.type = None
    #         else:
    #             axis_model.type = self.type.currentText()
    #     axis_model.start = empty_to_none(self.start.text())
    #     axis_model.stop = empty_to_none(self.stop.text())
    #     axis_model.num = empty_to_none(self.num.text())
    #     if self.are_points_editable():
    #         #axis_model.type = self.type.get
    #         axis_model.points = empty_to_none(self.points.toPlainText())
    #     else:
    #         axis_model.points = ''

    def _auto_enable_points(self):
        p = getattr(self, 'points', False)
        if p: p.setEnabled(self.are_points_editable())

    def fill_form(self):
        axis_model = getattr_by_path(self.controller.grid_model, self.axis_path)
        if self.allow_type_select:
            t = getattr(axis_model, 'type')
            with BlockQtSignals(self.type):
                if t is None:
                    self.type.setCurrentIndex(0)
                else:
                    self.type.setEditText(t)
        for attr_name in ('start', 'stop', 'num', 'points'):
            a = getattr(axis_model, attr_name)
            widget = getattr(self, attr_name, False)
            if widget:
                with BlockQtSignals(widget): widget.setText(none_to_empty(a))
        self._auto_enable_points()


class RectangularMesh1DController(GridController):
    """1D rectangular mesh (ordered or regular) script"""
    def __init__(self, document, model):
        super(RectangularMesh1DController, self).__init__(document=document, model=model)
        self.axis = AxisEdit(self, 'axis', None, allow_type_select=False, accept_non_regular=not model.is_regular)

    def fill_form(self):
         with self.mute_changes(): self.axis.fill_form()

    #def save_data_in_model(self):
    #    self.editor.to_model(self.model.axis)

    #def on_edit_enter(self):
    #    with self.mute_changes():
    #        self.editor.fill_form()

    def get_widget(self):
        return self.axis


class RectangularMeshController(GridController):
    """2D and 3D rectangular mesh script"""

    def __init__(self, document, model):
        super(RectangularMeshController, self).__init__(document=document, model=model)
        self.form = QGroupBox()
        vbox = QVBoxLayout()
        self.axis = []
        for i in range(0, model.dim):
            self.axis.append(
                AxisEdit(self, ('axis', i), model.axis_tag_name(i), title=AXIS_NAMES[model.dim - 1][i].title() + ' axis',
                         allow_type_select=True))
            vbox.addWidget(self.axis[-1])
        vbox.addStretch()
        self.form.setLayout(vbox)

    def fill_form(self):
        with self.mute_changes():
            for i in range(0, self.model.dim):
                self.axis[i].fill_form()

    # def save_data_in_model(self):
    #     for i in range(0, self.model.dim):
    #         self.axis_edit[i].to_model(self.model.axis[i])
    #
    # def on_edit_enter(self):
    #     with self.mute_changes():
    #         for i in range(0, self.model.dim):
    #             self.axis_edit[i].fill_form()

    def get_widget(self):
        return self.form
