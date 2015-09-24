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
from ..defines import get_defines_completer
from ...external.highlighter import SyntaxHighlighter, load_syntax
from ...external.highlighter.xml import syntax
from ...utils.textedit import TextEdit
from ...utils.widgets import VerticalScrollArea, EDITOR_FONT
from ...model.solvers.confsolver import Attr, AttrMulti, AttrChoice, AttrGeometryObject, AttrGeometryPath
from ..source import SCHEME


class SolverAutoWidget(VerticalScrollArea):

    def __init__(self, controller, parent=None):
        super(SolverAutoWidget, self).__init__(parent)

        self.controller = controller

        layout = QtGui.QFormLayout()
        layout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)

        label = QtGui.QLabel("General")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addRow(label)

        defines = get_defines_completer(self.controller.document.defines.model, self)

        self.geometry = QtGui.QComboBox()
        self.geometry.setEditable(True)
        self.geometry.textChanged.connect(self.controller.fire_changed)
        self.geometry.currentIndexChanged.connect(self.controller.fire_changed)
        self.geometry.setCompleter(defines)
        self.geometry.setToolTip('&lt;<b>geometry ref</b>=""&gt;<br/>'
                                 'Name of the existing geometry for use by this solver.')
        #TODO make sure the list is up-to date; add some graphical thumbnail
        layout.addRow("Geometry:", self.geometry)

        if controller.model.mesh_type is not None:
            self.mesh = QtGui.QComboBox()
            self.mesh.setEditable(True)
            self.mesh.textChanged.connect(self.controller.fire_changed)
            self.mesh.currentIndexChanged.connect(self.controller.fire_changed)
            self.mesh.setCompleter(defines)
            self.mesh.setToolTip('&lt;<b>mesh ref</b>=""&gt;<br/>'
                                 'Name of the existing {} mesh for use by this solver.'
                                 .format(controller.model.mesh_type))
            #TODO add some graphical thumbnail
            layout.addRow("Mesh:", self.mesh)
        else:
            self.mesh = None

        self.controls = {}

        for group, desc, items in controller.model.config:
            gname = group.split('/')[-1]
            label = QtGui.QLabel(desc)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            layout.addRow(label)
            if type(items) in (tuple, list):
                for item in items:
                    if isinstance(item, AttrChoice):
                        edit = QtGui.QComboBox()
                        edit.setEditable(True)
                        edit.addItems([''] + list(item.choices))
                        edit.textChanged.connect(self.controller.fire_changed)
                        edit.currentIndexChanged.connect(self.controller.fire_changed)
                        edit.setCompleter(defines)
                    elif isinstance(item, (AttrGeometryObject, AttrGeometryPath)):
                        edit = QtGui.QComboBox()
                        edit.setEditable(True)
                        edit.textChanged.connect(self.controller.fire_changed)
                        edit.currentIndexChanged.connect(self.controller.fire_changed)
                        edit.setCompleter(defines)
                    else:
                        if item.name[-1] == '#':
                            edit = QtGui.QPlainTextEdit()
                            edit.setFixedHeight(3 * edit.fontMetrics().lineSpacing())
                            edit.textChanged.connect(self.controller.fire_changed)
                        else:
                            edit = QtGui.QLineEdit()
                            edit.setCompleter(defines)
                            edit.textEdited.connect(self.controller.fire_changed)
                    edit.setToolTip(u'&lt;{} <b>{}</b>=""&gt;<br/>{}'.format(gname, item.name, item.help))
                    self.controls[group, item.name] = edit
                    layout.addRow(item.label + ':', edit)
            else:
                edit = TextEdit(parent, line_numbers=False)
                font = QtGui.QFont(EDITOR_FONT)
                font.setPointSize(font.pointSize()-1)
                edit.highlighter = SyntaxHighlighter(edit.document(), *load_syntax(syntax, SCHEME),
                                                     default_font=font)
                edit.setToolTip(u'&lt;<b>{0}</b>&gt;...&lt;/<b>{0}</b>&gt;<br/>{1}'.format(gname, desc))
                self.controls[group] = edit
                layout.addRow(edit)
                edit.textChanged.connect(self.controller.fire_changed)

        main = QtGui.QWidget()
        main.setLayout(layout)
        self.setWidget(main)

    def load_data(self):
        model = self.controller.model
        self.geometry.setCurrentIndex(self.geometry.findText(model.geometry))
        self.geometry.setEditText(model.geometry)
        if self.mesh is not None:
            self.mesh.setCurrentIndex(self.mesh.findText(model.mesh))
            self.mesh.setEditText(model.mesh)
        for group, _, items in model.config:
            if type(items) in (tuple, list):
                for item in items:
                    attr = item.name
                    edit = self.controls[group, attr]
                    if isinstance(item, AttrMulti):
                        attr = attr[:-1]
                        skip = len(attr)
                        data = model.data[group]
                        items = [(int(k[skip:]), data[k]) for k in data.keys() if k[:skip] == attr and k[-1].isdigit()]
                        if items:
                            values = (max(i[0] for i in items) + 1) * ['']
                            for i, v in items:
                                values[i] = v
                            edit.setPlainText('\n'.join(values))
                    else:
                        value = model.data[group][attr]
                        if isinstance(item, AttrGeometryObject):
                            edit.clear()
                            edit.addItems([''] + list(self.controller.document.geometry.model.names()))
                        if isinstance(item, AttrGeometryPath):
                            edit.clear()
                            edit.addItems([''] + list(self.controller.document.geometry.model.paths()))
                        if type(edit) == QtGui.QComboBox:
                            edit.setCurrentIndex(edit.findText(value))
                            edit.setEditText(value)
                        else:
                            edit.setText(value)
            else:
                edit = self.controls[group]
                edit.setPlainText(model.data[group])

    def save_data(self):
        model = self.controller.model
        model.geometry = self.geometry.currentText()
        if self.mesh is not None:
            model.mesh = self.mesh.currentText()
        for group, _, items in model.config:
            if type(items) in (tuple, list):
                for item in items:
                    attr = item.name
                    edit = self.controls[group, attr]
                    if isinstance(item, AttrMulti):
                        attr = attr[:-1]
                        values = edit.toPlainText().strip().splitlines()
                        for i,value in enumerate(values):
                            model.data[group][attr+str(i)] = value
                    else:
                        if type(edit) == QtGui.QComboBox:
                            model.data[group][attr] = edit.currentText()
                        else:
                            model.data[group][attr] = edit.text()
            else:
                edit = self.controls[group]
                model.data[group] = edit.toPlainText()