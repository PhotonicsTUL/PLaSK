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
from ...qt import QtGui
from ...utils.str import empty_to_none
from ...model.grids.mesh_rectilinear import AXIS_NAMES
from ..defines import get_defines_completer
from ...utils.widgets import ComboBox, TextEdit


class AxisEdit(QtGui.QGroupBox):

    def __init__(self, axis, controller, title=None, allow_type_select=False, accept_non_regular=False, axis_model=None):
        super(AxisEdit, self).__init__(title if title is not None else axis)
        if axis is None: axis = 'axis'
        defines = get_defines_completer(controller.document.defines.model, self)
        form_layout = QtGui.QFormLayout()
        self.allow_type_select = allow_type_select
        if not allow_type_select: self.accept_non_regular = accept_non_regular
        if self.allow_type_select:
            self.type = ComboBox()
            self.type.addItems(['(auto-detected)', 'ordered', 'regular'])
            self.type.setToolTip('&lt;{} <b>type</b>=""&gt;<br/>'
                                 'Type of axis. If auto-detected is selected, axis will be regular only if any of the '
                                 'start, stop or num attributes are specified (in other case it will be ordered).'
                                 .format(axis))
            self.type.editingFinished.connect(controller.fire_changed)
            self.type.currentIndexChanged.connect(controller.fire_changed)
            form_layout.addRow("Axis type:", self.type)
        self.start = QtGui.QLineEdit()
        self.start.setCompleter(defines)
        self.start.setToolTip(u'&lt;{} <b>start</b>="" stop="" num=""&gt;<br/>'
                              u'Position of the first point on the axis. (float [µm])'.format(axis))
        self.start.editingFinished.connect(controller.fire_changed)
        form_layout.addRow("Start:", self.start)
        self.stop = QtGui.QLineEdit()
        self.stop.setCompleter(defines)
        self.stop.setToolTip(u'&lt;{} start="" <b>stop</b>="" num=""&gt;\n'
                             u'Position of the last point on the axis. (float [µm])'.format(axis))
        self.stop.editingFinished.connect(controller.fire_changed)
        form_layout.addRow("Stop:", self.stop)
        self.num = QtGui.QLineEdit()
        self.num.setCompleter(defines)
        self.num.setToolTip('&lt;{} start="" stop="" <b>num</b>=""&gt;<br/>'
                            'Number of the equally distributed points along the axis. (integer)'.format(axis))
        self.num.editingFinished.connect(controller.fire_changed)
        form_layout.addRow("Num:", self.num)
        if allow_type_select or accept_non_regular:
            self.points = TextEdit()
            self.points.setToolTip('&lt;{0}&gt;<b><i>points</i></b>&lt;/{0}&gt;<br/>'
                                   'Comma-separated list of the mesh points along this axis.'.format(axis))
            # self.points.editingFinished.connect(controller.fire_changed)
            #self.points.setWordWrapMode(QtGui.QTextEdit.LineWrapMode)
            form_layout.addRow("Points:", self.points)
            if allow_type_select:
                self.points.setVisible(self.are_points_editable())
        self.setLayout(form_layout)
        if axis_model is not None: self.from_model(axis_model)

    def are_points_editable(self):
        if self.allow_type_select:
            return self.type.currentText() != 'regular'
        else:
            return self.accept_non_regular

    def to_model(self, axis_model):
        if self.allow_type_select:
            if self.type.currentIndex() == 0:
                axis_model.type = None
            else:
                axis_model.type = self.type.currentText()
        axis_model.start = empty_to_none(self.start.text())
        axis_model.stop = empty_to_none(self.stop.text())
        axis_model.num = empty_to_none(self.num.text())
        if self.are_points_editable():
            #axis_model.type = self.type.get
            axis_model.points = empty_to_none(self.points.toPlainText())
        else:
            axis_model.points = ''

    def from_model(self, axis_model):
        if self.allow_type_select:
            t = getattr(axis_model, 'type')
            if t is None:
                self.type.setCurrentIndex(0)
            else:
                self.type.setEditText(t)
        for attr_name in ('start', 'stop', 'num', 'points'):
            a = getattr(axis_model, attr_name)
            widget = getattr(self, attr_name, False)
            if widget: widget.setText('' if a is None else a)


class RectangularMesh1DController(GridController):
    """1D rectangular mesh (ordered or regular) script"""
    def __init__(self, document, model):
        super(RectangularMesh1DController, self).__init__(document=document, model=model)
        self.editor = AxisEdit(None, self, allow_type_select=False, accept_non_regular=not model.is_regular)

    def save_data_in_model(self):
        self.editor.to_model(self.model.axis)

    def on_edit_enter(self):
        with self.mute_changes():
            self.editor.from_model(self.model.axis)

    def get_widget(self):
        return self.editor


class RectangularMeshController(GridController):
    """2D and 3D rectangular mesh script"""

    def __init__(self, document, model):
        super(RectangularMeshController, self).__init__(document=document, model=model)
        self.form = QtGui.QGroupBox()
        vbox = QtGui.QVBoxLayout()
        self.axis_edit = []
        for i in range(0, model.dim):
            self.axis_edit.append(AxisEdit(model.axis_tag_name(i), self,
                                           title=AXIS_NAMES[model.dim-1][i].title() + ' axis',
                                           allow_type_select=True))
            vbox.addWidget(self.axis_edit[-1])
        vbox.addStretch()
        self.form.setLayout(vbox)

    def save_data_in_model(self):
        for i in range(0, self.model.dim):
            self.axis_edit[i].to_model(self.model.axis[i])

    def on_edit_enter(self):
        with self.mute_changes():
            for i in range(0, self.model.dim):
                self.axis_edit[i].from_model(self.model.axis[i])

    def get_widget(self):
        return self.form
