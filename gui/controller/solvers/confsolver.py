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
from ...utils.texteditor import TextEditor
from ...utils.widgets import VerticalScrollArea, EDITOR_FONT, TextEditWithCB
from ...model.solvers.confsolver import Attr, AttrMulti, AttrChoice, AttrGeometryObject, AttrGeometryPath
from ..source import SCHEME
from . import Controller


class SolverAutoWidget(VerticalScrollArea):

    class ChangeItemCommand(QtGui.QUndoCommand):

        def __init__(self, model, node, setter, new_value, old_value, action_name, QUndoCommand_parent = None):
            super(SolverAutoWidget.ChangeItemCommand, self).__init__(action_name, QUndoCommand_parent)
            self.model = model
            self.node = node
            self.setter = setter
            self.new_value = new_value
            self.old_value = old_value

        def set_property_value(self, value):
            self.setter(self.node, value)
            #index = self.model.entries.index(self.node)
            #self.model.dataChanged.emit(index, index)
            self.model.fire_changed()

        def redo(self):
            self.set_property_value(self.new_value)

        def undo(self):
            self.set_property_value(self.old_value)

    def _change_attr(self, group, attr, value):
        def set_solver_attr(node, value): node.data[group][attr] = value
        node = self.controller.solver_model
        model = self.controller.section_model
        old_value = node.data[group][attr]
        if value != old_value:
            model.undo_stack.push(SolverAutoWidget.ChangeItemCommand(
                model, node, set_solver_attr, value, old_value, 'change attribute of solver'
            ))


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
                            edit = TextEditWithCB()
                            edit.setFixedHeight(3 * edit.fontMetrics().lineSpacing())
                            edit.textChanged.connect(self.controller.fire_changed)
                            #TODO see save_data
                            #edit.focus_out_cb = lambda edit=edit, group=group, name=item.name: self._change_attr(group, name, edit.text())
                        else:
                            edit = QtGui.QLineEdit()
                            edit.setCompleter(defines)
                            edit.textEdited.connect(self.controller.fire_changed)
                            #edit.editingFinished.connect(lambda edit=edit, group=group, name=item.name:
                            #                             self._change_attr(group, name, edit.text()))

                    edit.setToolTip(u'&lt;{} <b>{}</b>=""&gt;<br/>{}'.format(gname, item.name, item.help))
                    self.controls[group, item.name] = edit
                    layout.addRow(item.label + ':', edit)
            else:
                edit = TextEditor(parent, line_numbers=False)
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


class ConfSolverController(Controller):
    """Class for solvers defined in configuration dictionary"""

    """
        :param document:
        :param model: model of solver to configure TODO should be section model?
    """
    def __init__(self, document, model):
        super(ConfSolverController, self).__init__(document, model)
        try:
            widget_class = self.model.widget
            if widget_class is None: raise AttributeError
        except AttributeError:
            widget_class = SolverAutoWidget
        self.widget = widget_class(self)
        #self.section_model.changed.connect(self._model_change_cb)

    def _model_change_cb(self, *args, **kwargs):
        self.widget.load_data()

    """
        :return ConfSolver: model of edited solver
    """
    @property
    def solver_model(self):
        return self.model

    """
        :return SolversModel: model of a whole solver's section
    """
    @property
    def section_model(self):
        return self.solver_model.tree_parent

    def get_widget(self):
        return self.widget

    def on_edit_enter(self):
        with self.mute_changes():
            try:
                if self.model.solver.endswith('2D'):
                    geometries = [g.name for g in self.document.geometry.model.roots_cartesian2d if g.name]
                elif self.model.solver.endswith('Cyl'):
                    geometries = [g.name for g in self.document.geometry.model.roots_cylindrical if g.name]
                elif self.model.solver.endswith('3D'):
                    geometries = [g.name for g in self.document.geometry.model.roots_cartesian3d if g.name]
                else:
                    raise AttributeError
            except AttributeError:
                pass
            else:
                self.widget.geometry.clear()
                self.widget.geometry.addItems([''] + geometries)
            try:
                grids = []
                if self.model.mesh_type is not None:
                    if ',' in self.model.mesh_type:
                        mesh_types = (m.strip() for m in self.model.mesh_type.split(','))
                    else:
                        mesh_types = self.model.mesh_type,
                    for mesh_type in mesh_types:
                        mesh_type = mesh_type.lower()
                        grids.extend(m.name for m in self.document.grids.model.entries if m.name and m.type == mesh_type)
            except AttributeError:
                pass
            else:
                if self.widget.mesh is not None:
                    self.widget.mesh.clear()
                    self.widget.mesh.addItems([''] + grids)
            self.widget.load_data()

    def save_data_in_model(self):
        self.widget.save_data()