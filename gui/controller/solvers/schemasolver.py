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
from copy import deepcopy

from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from ..defines import get_defines_completer
from ...lib.highlighter import SyntaxHighlighter, load_syntax
from ...lib.highlighter.xml import syntax
from ...utils.str import empty_to_none
from ...utils.texteditor import TextEditorWithCB
from ...utils.widgets import VerticalScrollArea, EDITOR_FONT, ComboBox, MultiLineEdit
from ...utils.qsignals import BlockQtSignals
from ...utils.qundo import UndoCommandWithSetter
from ...model.solvers.schemasolver import SchemaTag, \
    AttrGroup, AttrMulti, AttrChoice, AttrGeometryObject, AttrGeometryPath, AttrGeometry, AttrMesh
from ...model.solvers.bconds import SchemaBoundaryConditions
from ..source import SCHEME
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
        color = QPalette().color(QPalette.Normal, QPalette.Window if conflict else QPalette.Base)
        palette = widget.palette()
        palette.setColor(QPalette.Base, color)
        widget.setPalette(palette)
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


class SolverWidget(VerticalScrollArea):

    def _change_attr(self, group, name, value, attr=None):
        node = self.controller.solver_model

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
        if isinstance(attr, AttrChoice):
            edit = ComboBox()
            edit.setEditable(True)
            edit.addItems([''] + list(attr.choices))
            edit.editingFinished.connect(lambda edit=edit, group=group, name=attr.name, attr=attr:
                                         self._change_attr(group, name, edit.currentText(), attr))
            completer = get_defines_completer(defines, edit, strings=attr.choices)
            edit.setCompleter(completer)
            if attr.default is not None:
                edit.lineEdit().setPlaceholderText(attr.default)
        elif isinstance(attr, (AttrGeometryObject, AttrGeometryPath, AttrGeometry, AttrMesh)):
            edit = ComboBox()
            edit.setEditable(True)
            edit.editingFinished.connect(lambda edit=edit, group=group, name=attr.name:
                                         self._change_attr(group, name, edit.currentText()))
            edit.setCompleter(get_defines_completer(defines, edit))
            if attr.default is not None:
                edit.lineEdit().setPlaceholderText(attr.default)
        else:
            if attr.name[-1] == '#':
                edit = MultiLineEdit(movable=True, document=self.controller.document)
                # edit.setFixedHeight(3 * edit.fontMetrics().lineSpacing())
                # edit.textChanged.connect(self.controller.fire_changed)
                edit.change_cb = lambda edit=edit, group=group, name=attr.name, attr=attr: \
                    self._change_multi_attr(group, name, edit.get_values(), attr)
            else:
                edit = QLineEdit()
                edit.setCompleter(defines)
                # edit.textEdited.connect(self.controller.fire_changed)
                edit.editingFinished.connect(lambda edit=edit, group=group, name=attr.name, attr=attr:
                                             self._change_attr(group, name, edit.text(), attr))
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
        button.setArrowType(Qt.DownArrow)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setIconSize(QSize(8, 8))
        button.setStyleSheet("""
            border: none;
            margin-left: -2px;
            padding-left: 0px;
        """)
        button.setText(header)

        def toggled(selected):
            button.setArrowType(Qt.DownArrow if selected else Qt.RightArrow)
            for edit in rows:
                edit.setVisible(selected)

        button.toggled.connect(toggled)

        font = button.font()
        font.setBold(True)
        button.setFont(font)

        self.form_layout.addRow(button)

    def _add_row(self, label, edit, rows):
        rows.append(edit)
        self.form_layout.addRow(label + ':', edit)
        label_widget = self.form_layout.labelForField(edit)
        rows.append(label_widget)

    def __init__(self, controller, parent=None):
        super(SolverWidget, self).__init__(parent)

        self.controller = controller

        self.form_layout = QFormLayout()
        self.form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        main = QWidget()

        rows = []
        if controller.model.geometry_type or controller.model.mesh_types:
            self._make_header("General", rows)

        defines = get_defines_completer(self.controller.document.defines.model, self)

        if controller.model.geometry_type is not None:
            self.geometry = ComboBox()
            self.geometry.setEditable(True)
            self.geometry.editingFinished.connect(
                lambda w=self.geometry: self._change_node_field('geometry', w.currentText()))
            self.geometry.setCompleter(get_defines_completer(defines, self.geometry))
            self.geometry.setToolTip(u'&lt;<b>geometry ref</b>=""&gt;<br/>'
                                     u'Name of the existing geometry for use by this solver.')
            # TODO add some graphical thumbnail
            self._add_row("Geometry", self.geometry, rows)
        else:
            self.geometry = None

        if controller.model.mesh_types:
            self.mesh = ComboBox()
            self.mesh.setEditable(True)
            self.mesh.editingFinished.connect(lambda w=self.mesh: self._change_node_field('mesh', w.currentText()))
            self.mesh.setCompleter(get_defines_completer(defines, self.geometry))
            self.mesh.setToolTip(u'&lt;<b>mesh ref</b>=""&gt;<br/>'
                                 u'Name of the existing {} mesh for use by this solver.'
                                 .format(' or '.join(controller.model.mesh_types)))
            # TODO add some graphical thumbnail
            self._add_row("Mesh", self.mesh, rows)
        else:
            self.mesh = None

        self.controls = {}

        last_header = None
        for schema in controller.model.schema:
            group = schema.name
            gname = group.split('/')[-1]
            bc = isinstance(schema, SchemaBoundaryConditions)
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
                            field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
                            label = QLabel(sep + item.label + ':')
                            lay.addWidget(label)
                            lay.addWidget(field)
                            sep = ' '
                    else:
                        edit = self._add_attr(attr, defines, gname, group)
                    self._add_row(attr.label, edit, rows)
            elif bc:
                edit = QPushButton("View / Edit")
                edit.sizePolicy().setHorizontalStretch(1)
                edit.pressed.connect(lambda schema=schema: self.edit_boundary_conditions(schema))
                self.controls[group] = edit
                self._add_row(schema.label2, edit, rows)
            else:
                edit = TextEditorWithCB(parent=parent, line_numbers=False)
                font = QFont(EDITOR_FONT)
                font.setPointSize(font.pointSize() - 1)
                edit.highlighter = SyntaxHighlighter(edit.document(), *load_syntax(syntax, SCHEME),
                                                     default_font=font)
                edit.setToolTip(u'&lt;<b>{0}</b>&gt;...&lt;/<b>{0}</b>&gt;<br/>{1}'.format(gname, schema.label))
                self.controls[group] = edit
                rows.append(edit)
                self.form_layout.addRow(edit)
                # edit.textChanged.connect(self.controller.fire_changed)
                edit.focus_out_cb = lambda edit=edit, group=group: self._change_attr(group, None, edit.toPlainText())

        main.setLayout(self.form_layout)
        self.setWidget(main)

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
                                    self._set_items(edit, list(self.controller.document.geometry.model.names()))
                                except AttributeError:
                                    pass
                            elif isinstance(item, AttrGeometryPath):
                                try:
                                    self._set_items(edit, list(self.controller.document.geometry.model.paths()))
                                except AttributeError:
                                    pass
                            elif isinstance(item, AttrMesh):
                                try:
                                    self._set_items(edit, list(self._get_grids(item.types)))
                                except AttributeError:
                                    pass
                            if type(edit) in (ComboBox, QComboBox):
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
            else:
                edit = self.controls[group]
                with BlockQtSignals(edit):
                    edit.setPlainText(model.data[group])

    def edit_boundary_conditions(self, schema):
        data = deepcopy(self.controller.model.data[schema.name])
        dialog = BoundaryConditionsDialog(self.controller, schema, data)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            self._change_boundary_condition(schema, data)


class SchemaSolverController(Controller):
    """
    Class for solvers defined in configuration dictionary
        :param document:
        :param model: model of solver to configure TODO should be section model?
    """

    def __init__(self, document, model):
        super(SchemaSolverController, self).__init__(document, model)
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
        if info.what == 'geometry' and self.widget.geometry is not None:
            self.widget.geometry.setFocus()
        elif info.what == 'mesh' and self.widget.mesh is not None:
            self.widget.mesh.setFocus()
        elif isinstance(info.what, tuple) and info.what in self.widget.controls:
            self.widget.controls[info.what].setFocus()
