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
from copy import deepcopy

from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from ...qt import qt_exec
from ..defines import get_defines_completer
from ...lib.highlighter import SyntaxHighlighter, load_syntax
from ...lib.highlighter.xml import SYNTAX
from ...utils.str import empty_to_none
from ...utils.texteditor import TextEditorWithCB
from ...utils.widgets import VerticalScrollArea, EDITOR_FONT, EditComboBox, MultiLineEdit, LineEditWithClear
from ...utils.qsignals import BlockQtSignals
from ...utils.qundo import UndoCommandWithSetter
from ...model.solvers.schemasolver import SchemaTag, SchemaCustomWidgetTag, \
    AttrGroup, AttrMulti, AttrChoice, AttrGeometryObject, AttrGeometryPath, AttrGeometry, AttrMesh
from ...model.solvers.bconds import SchemaBoundaryConditions
from ...utils.texteditor.xml import XML_SCHEME
from . import Controller
from .bconds import BoundaryConditionsDialog


def get_attr_list(model, group, attr):
    attr = attr[:-1]
    skip = len(attr)
    data = model.data[group]
    items = [(int(k[skip:]), data[k]) for k in data.keys() if k[:skip] == attr and k[-1].isdigit()]
    if items:
        values = (max(i[0] for i in items) + 1) * ['']
        for i, v in items:
            values[i] = v
        return values
    return []


def set_attr_list(model, group, attr, values):
    attr = attr[:-1]
    data = model.data[group]

    skip = len(attr)  # delete old attributes (data can have more attributes than values)
    for k in list(data.keys()):
        if k[:skip] == attr and k[-1].isdigit(): del data[k]

    for i, value in enumerate(values):
        if value != '':
            data[attr + str(i)] = value


def set_conflict(widget, conflict):
    with BlockQtSignals(widget):
        widget.setEnabled(not conflict)
        color = QPalette().color(QPalette.ColorGroup.Normal, QPalette.ColorRole.Window if conflict else QPalette.ColorRole.Base)
        # palette = widget.palette()
        # palette.setColor(QPalette.ColorRole.Base, color)
        # widget.setPalette(palette)
        try:
            if conflict:
                placeholder = widget.placeholderText()
                if placeholder:
                    widget._placeholder = placeholder
                widget.setPlaceholderText("")
            else:
                widget.setPlaceholderText(widget._placeholder)
        except AttributeError:
            pass


class SolverWidget(QWidget):

    def __init__(self, controller, parent=None):
        super().__init__(parent)

        self.controller = weakref.proxy(controller)

        scroll = VerticalScrollArea(self)

        self.filter = LineEditWithClear()
        self.filter.setPlaceholderText("Filter...")
        self.filter.textChanged.connect(self.build_form)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.filter)
        layout.addWidget(scroll)
        layout.setSpacing(3)
        self.setLayout(layout)

        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        main = QWidget()

        self.headers = []

        defines = get_defines_completer(self.controller.document.defines.model, self)

        last_header = None

        weakself = weakref.proxy(self)

        if controller.model.geometry_type is not None:
            rows = []
            last_header = "Geometry"
            self._make_header(last_header, rows)
            self.geometry = EditComboBox()
            self.geometry.setEditable(True)
            self.geometry.editingFinished.connect(
                lambda w=self.geometry: weakself._change_node_field('geometry', w.currentText()))
            self.geometry.setCompleter(get_defines_completer(defines, self.geometry))
            self.geometry.setToolTip(u'&lt;<b>geometry ref</b>=""&gt;<br/>'
                                     u'Name of the existing geometry for use by this solver.')
            # TODO add some graphical thumbnail
            self._add_row(rows, self.geometry, "Geometry")
        else:
            self.geometry = None

        if controller.model.mesh_types:
            rows = []
            last_header = "Mesh"
            self._make_header(last_header, rows)
            self.mesh = EditComboBox()
            self.mesh.setEditable(True)
            self.mesh.editingFinished.connect(lambda w=self.mesh: weakself._change_node_field('mesh', w.currentText()))
            self.mesh.setCompleter(get_defines_completer(defines, self.geometry))
            self.mesh.setToolTip(u'&lt;<b>mesh ref</b>=""&gt;<br/>'
                                 u'Name of the existing {} mesh for use by this solver.'
                                 .format(' or '.join(controller.model.mesh_types)))
            # TODO add some graphical thumbnail

            self._add_row(rows, self.mesh, "Mesh")
        else:
            self.mesh = None

        self.controls = {}

        for schema in controller.model.schema:
            group = schema.name
            gname = group.split('/')[-1]
            if last_header != schema.label:
                last_header = schema.label
                rows = []
                self._make_header(last_header, rows)
            if isinstance(schema, SchemaTag):
                for attr in schema.attrs:
                    if isinstance(attr, AttrGroup):
                        edit = QWidget()
                        lay = QHBoxLayout()
                        lay.setContentsMargins(0, 0, 0, 0)
                        edit.setLayout(lay)
                        sep = ''
                        for item in attr:
                            field = self._add_attr(item, defines, gname, group)
                            field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
                            label = QLabel(sep + item.label + ':')
                            lay.addWidget(label)
                            lay.addWidget(field)
                            sep = ' '
                    else:
                        edit = self._add_attr(attr, defines, gname, group)
                    self._add_row(rows, edit, attr.label)
            elif isinstance(schema, SchemaCustomWidgetTag):
                edit = QPushButton(schema.button_label)
                edit.sizePolicy().setHorizontalStretch(1)
                edit.pressed.connect(lambda schema=schema: weakself.launch_custom_editor(schema))
                self.controls[group] = edit
                self._add_row(rows, edit, schema.label)
            elif isinstance(schema, SchemaBoundaryConditions):
                edit = QPushButton("View / Edit")
                edit.sizePolicy().setHorizontalStretch(1)
                edit.pressed.connect(lambda schema=schema: weakself.edit_boundary_conditions(schema))
                self.controls[group] = edit
                self._add_row(rows, edit, schema.label2)
            else:
                edit = TextEditorWithCB(parent=parent, line_numbers=False)
                font = QFont(EDITOR_FONT)
                font.setPointSize(font.pointSize() - 1)
                edit.highlighter = SyntaxHighlighter(edit.document(), *load_syntax(SYNTAX, XML_SCHEME),
                                                     default_font=font)
                edit.setToolTip(u'&lt;<b>{0}</b>&gt;...&lt;/<b>{0}</b>&gt;<br/>{1}'.format(gname, schema.label))
                self.controls[group] = edit
                self._add_row(rows, edit)
                # edit.textChanged.connect(self.controller.fire_changed)
                edit.focus_out_cb = lambda edit=edit, group=group: weakself._change_attr(group, None, edit.toPlainText())

        self.build_form()

        main.setLayout(self.form_layout)
        scroll.setWidget(main)

    def _change_attr(self, group, name, value, attr=None):
        try:
            node = self.controller.solver_model
        except ReferenceError:
            # this can happen when closing window, so we have not choice but to ignore it
            return

        def set_solver_attr(value):
            if name is None:
                node.data[group] = value
            else:
                node.data[group][name] = value

        model = self.controller.section_model
        old_value = node.data[group] if name is None else node.data[group][name]
        value = empty_to_none(value)
        if value != old_value:
            model.undo_stack.push(UndoCommandWithSetter(
                model, set_solver_attr, value, old_value,
                u"change solver's {}".format('attribute' if attr is None else attr.label.strip())
            ))

    def _change_multi_attr(self, group, name, values, attr=None):
        node = self.controller.solver_model

        def set_solver_attr(vals):
            set_attr_list(node, group, name, vals)

        model = self.controller.section_model
        old_values = get_attr_list(node, group, name)
        if values != old_values:
            model.undo_stack.push(UndoCommandWithSetter(
                model, set_solver_attr, values, old_values,
                u"change solver's {}".format('attribute' if attr is None else attr.label.strip())
            ))

    def _change_node_field(self, field_name, value):
        node = self.controller.solver_model

        def set_solver_field(value):
            setattr(node, field_name, value)

        model = self.controller.section_model
        old_value = getattr(node, field_name)
        value = empty_to_none(value)
        if value != old_value:
            model.undo_stack.push(UndoCommandWithSetter(
                model, set_solver_field, value, old_value, "change solver's {}".format(field_name)
            ))

    def _change_boundary_condition(self, schema, data):
        node = self.controller.solver_model

        def set_value(value):
            node.data[schema.name] = value
            self.controls[schema.name].setText("     View / Edit  ({})".format(len(value)))

        model = self.controller.section_model
        old_data = node.data[schema.name]
        if data != old_data:
            model.undo_stack.push(UndoCommandWithSetter(
                model, set_value, data, old_data,
                u"change solver's {}"
                    .format(schema.name if schema.label is None else schema.label.strip().lower())
            ))

    def _add_attr(self, attr, defines, gname, group):
        weakself = weakref.proxy(self)
        if isinstance(attr, AttrChoice):
            edit = EditComboBox()
            edit.setEditable(True)
            edit.addItems([''] + list(attr.choices))
            edit.editingFinished.connect(lambda edit=edit, group=group, name=attr.name, attr=attr:
                                         weakself._change_attr(group, name, edit.currentText(), attr))
            completer = get_defines_completer(defines, edit, strings=attr.choices)
            edit.setCompleter(completer)
            if attr.default is not None:
                edit.lineEdit().setPlaceholderText(attr.default)
        elif isinstance(attr, (AttrGeometryObject, AttrGeometryPath, AttrGeometry, AttrMesh)):
            edit = EditComboBox()
            edit.setEditable(True)
            edit.editingFinished.connect(lambda edit=edit, group=group, name=attr.name:
                                         weakself._change_attr(group, name, edit.currentText()))
            edit.setCompleter(get_defines_completer(defines, edit))
            if attr.default is not None:
                edit.lineEdit().setPlaceholderText(attr.default)
        else:
            if attr.name[-1] == '#':
                edit = MultiLineEdit(movable=True, document=self.controller.document)
                # edit.setFixedHeight(3 * edit.fontMetrics().lineSpacing())
                # edit.textChanged.connect(self.controller.fire_changed)
                edit.change_cb = lambda edit=edit, group=group, name=attr.name, attr=attr: \
                    weakself._change_multi_attr(group, name, edit.get_values(), attr)
            else:
                edit = QLineEdit()
                edit.setCompleter(defines)
                # edit.textEdited.connect(self.controller.fire_changed)
                edit.editingFinished.connect(lambda edit=edit, group=group, name=attr.name, attr=attr:
                                             weakself._change_attr(group, name, edit.text(), attr))
                if attr.default is not None:
                    edit.setPlaceholderText(attr.default)
        edit.setToolTip(u'&lt;{} <b>{}</b>="{}"&gt;{}<br/>{}'.format(
            gname, attr.name, '' if attr.default is None else attr.default,
            " ({})".format(attr.typ) if attr.typ else "", attr.help.replace('\n', '<br/><br/>')))
        self.controls[group, attr.name] = edit
        return edit

    def _make_header(self, header, rows):
        button = QToolButton()
        button.setCheckable(True)
        button.setChecked(True)
        button.setArrowType(Qt.ArrowType.DownArrow)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setIconSize(QSize(8, 8))
        button.setStyleSheet("""
            border: none;
            margin-left: -2px;
            padding-left: 0px;
        """)
        button.setText(header)

        weakself = weakref.proxy(self)

        def toggled(selected):
            button.setArrowType(Qt.ArrowType.DownArrow if selected else Qt.ArrowType.RightArrow)
            self.build_form()

        button.toggled.connect(toggled)

        font = button.font()
        font.setBold(True)
        button.setFont(font)

        self.headers.append((button, rows))

    def build_form(self, text=None):
        if text is None:
            text = self.filter.text()
        for _ in range(self.form_layout.rowCount()):
            item = self.form_layout.itemAt(0, QFormLayout.ItemRole.FieldRole)
            widget = item.layout() or item.widget()
            widget.setParent(None)
            self.form_layout.removeRow(0)
        if text:
            text = text.lower()
            for button, rows in self.headers:
                self.form_layout.addRow(button)
                if button.isChecked():
                    if text in button.text().lower():
                        for row in rows:
                            self.form_layout.addRow(*row)
                    else:
                        for row in rows:
                            if text in row[0].lower():
                                self.form_layout.addRow(*row)
        else:
            for button, rows in self.headers:
                self.form_layout.addRow(button)
                if button.isChecked():
                    for row in rows:
                        self.form_layout.addRow(*row)

    def _add_row(self, rows, edit, label=None):
        if label is not None:
            rows.append((label + ':', edit))
        else:
            rows.append((edit,))

    def _get_grids(self, mesh_types):
        if mesh_types is None:
            grids = [m.name for m in self.controller.document.grids.model.entries if m.name]
        else:
            grids = []
            for mesh_type in mesh_types:
                mesh_type = mesh_type.lower()
                grids.extend(m.name for m in self.controller.document.grids.model.entries
                             if m.name and m.type == mesh_type)
        return grids

    @staticmethod
    def _set_items(edit, strings):
        edit.clear()
        edit.addItems([''] + strings)
        edit.completer().model().strings = strings

    def load_data(self):
        model = self.controller.model
        if self.geometry is not None:
            with BlockQtSignals(self.geometry):
                try:
                    geometries = [g.name for g in getattr(self.controller.document.geometry.model,
                                                          "roots_" + model.geometry_type.lower())
                                  if g.name]
                except AttributeError:
                    pass
                else:
                    self._set_items(self.geometry, geometries)
                self.geometry.setCurrentIndex(self.geometry.findText(model.geometry))
                self.geometry.setEditText(model.geometry)
        if self.mesh is not None:
            with BlockQtSignals(self.mesh):
                try:
                    grids = self._get_grids(model.mesh_types)
                except AttributeError:
                    pass
                else:
                    if self.mesh is not None:
                        self._set_items(self.mesh, grids)
                self.mesh.setCurrentIndex(self.mesh.findText(model.mesh))
                self.mesh.setEditText(model.mesh)
        for edit in self.controls.values():
            set_conflict(edit, False)
        for schema in model.schema:
            group = schema.name
            if isinstance(schema, SchemaTag):
                for item in schema.attrs.flat:
                    attr = item.name
                    edit = self.controls[group, attr]
                    with BlockQtSignals(edit):
                        if isinstance(item, AttrMulti):
                            value = get_attr_list(model, group, attr)
                            edit.set_values(value)
                        else:
                            value = model.data[group][attr]
                            if isinstance(item, AttrGeometry):
                                try:
                                    self._set_items(edit,
                                                    [g.name for g in getattr(self.controller.document.geometry.model,
                                                                             "roots_" + item.type)
                                                     if g.name])
                                except AttributeError:
                                    pass
                            elif isinstance(item, AttrGeometryObject):
                                try:
                                    self._set_items(edit, list(self.controller.document.geometry.model.get_names()))
                                except AttributeError:
                                    pass
                            elif isinstance(item, AttrGeometryPath):
                                try:
                                    self._set_items(edit, list(self.controller.document.geometry.model.get_paths()))
                                except AttributeError:
                                    pass
                            elif isinstance(item, AttrMesh):
                                try:
                                    self._set_items(edit, list(self._get_grids(item.types)))
                                except AttributeError:
                                    pass
                            if isinstance(edit, QComboBox):
                                edit.setCurrentIndex(edit.findText(value))
                                edit.setEditText(value)
                            else:
                                edit.setText(value)
                        if value:
                            set_conflict(edit, False)
                            for conflict in item.conflicts:
                                conflicting = self.controls.get(conflict)
                                if conflicting is not None:
                                    set_conflict(conflicting, True)
            elif isinstance(schema, SchemaBoundaryConditions):
                self.controls[group].setText("     View / Edit  ({})".format(len(model.data[group])))
            elif not isinstance(schema, SchemaCustomWidgetTag):
                edit = self.controls[group]
                with BlockQtSignals(edit):
                    edit.setPlainText(model.data[group])

    def edit_boundary_conditions(self, schema):
        data = deepcopy(self.controller.model.data[schema.name])
        dialog = BoundaryConditionsDialog(self.controller, schema, data)
        result = qt_exec(dialog)
        if result == QDialog.DialogCode.Accepted:
            self._change_boundary_condition(schema, data)

    def launch_custom_editor(self, schema):
        old_data = self.controller.model.data[schema.name]
        new_data = schema.edit_func(deepcopy(old_data), self.controller.document)
        if new_data is not None and new_data != old_data:
            assert new_data.name == schema.name
            node = self.controller.solver_model
            def set_value(value):
                node.data[schema.name] = value
            model = self.controller.section_model
            model.undo_stack.push(UndoCommandWithSetter(
                model, set_value, new_data, old_data,
                u"change solver's {}"
                    .format(schema.name if schema.label is None else schema.label.strip().lower())
            ))


class SchemaSolverController(Controller):
    """
    Class for solvers defined in configuration dictionary
        :param document:
        :param model: model of solver to configure TODO should be section model?
    """

    def __init__(self, document, model):
        super().__init__(document, model)
        try:
            widget_class = self.model.widget
            if widget_class is None: raise AttributeError
        except AttributeError:
            widget_class = SolverWidget
        self.widget = widget_class(self)
        self.section_model.changed.connect(self._model_change_cb)

    def _model_change_cb(self, *args, **kwargs):
        self.widget.load_data()

    @property
    def solver_model(self):
        """
            :return SchemaSolver: model of edited solver
        """
        return self.model

    @property
    def section_model(self):
        """
            :return SolversModel: model of a whole solver's section
        """
        return self.solver_model.tree_parent

    def get_widget(self):
        return self.widget

    def on_edit_enter(self):
        with self.mute_changes():
            self.widget.load_data()

    def select_info(self, info):
        what = getattr(info, 'what', None)
        if what == 'geometry' and self.widget.geometry is not None:
            self.widget.geometry.setFocus()
        elif what == 'mesh' and self.widget.mesh is not None:
            self.widget.mesh.setFocus()
        elif isinstance(what, tuple) and what in self.widget.controls:
            self.widget.controls[what].setFocus()
