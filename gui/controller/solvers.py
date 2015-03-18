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

from ..qt import QtCore, QtGui
from ..qt.QtGui import QSplitter, QItemSelectionModel
from ..qt.QtCore import Qt

from ..model.connects import PROPS

from ..utils.widgets import table_last_col_fill, table_edit_shortcut, VerticalScrollArea
from ..utils.textedit import TextEdit
from ..utils.widgets import DEFAULT_FONT
from ..external.highlighter import SyntaxHighlighter, load_syntax
from . import Controller
from .table import table_with_manipulators
from .source import scheme, syntax
from .defines import get_defines_completer


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
                    if len(item) == 4:
                        try:
                            attr, text, help, choices = item
                        except ValueError:
                            import sys
                            sys.stderr.write(str(item)+'\n')
                            raise
                        edit = QtGui.QComboBox()
                        edit.setEditable(True)
                        edit.addItems([''] + list(choices))
                        edit.textChanged.connect(self.controller.fire_changed)
                        edit.currentIndexChanged.connect(self.controller.fire_changed)
                        edit.setCompleter(defines)
                    else:
                        try:
                            attr, text, help = item
                        except ValueError:
                            import sys
                            sys.stderr.write(str(item)+'\n')
                            raise
                        if attr[-1] == '#':
                            edit = QtGui.QPlainTextEdit()
                            edit.setFixedHeight(3 * edit.fontMetrics().lineSpacing())
                            edit.textChanged.connect(self.controller.fire_changed)
                        else:
                            edit = QtGui.QLineEdit()
                            edit.setCompleter(defines)
                            edit.textEdited.connect(self.controller.fire_changed)
                    edit.setToolTip(u'&lt;{} <b>{}</b>=""&gt;<br/>{}'.format(gname, attr, help))
                    self.controls[group, attr] = edit
                    layout.addRow(text + ':', edit)
            else:
                edit = TextEdit(parent, line_numbers=False)
                font = QtGui.QFont(DEFAULT_FONT)
                font.setPointSize(font.pointSize()-1)
                edit.highlighter = SyntaxHighlighter(edit.document(), *load_syntax(syntax, scheme),
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
                    attr = item[0]
                    edit = self.controls[group, attr]
                    if attr[-1] == '#':
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
                    attr = item[0]
                    edit = self.controls[group, attr]
                    if attr[-1] == '#':
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

    def __init__(self, document, model):
        super(ConfSolverController, self).__init__(document, model)
        try:
            widget_class = self.model.widget
            if widget_class is None: raise AttributeError
        except AttributeError:
            widget_class = SolverAutoWidget
        self.widget = widget_class(self)

    def get_widget(self):
        return self.widget

    def on_edit_enter(self):
        self.notify_changes = False
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
            mesh_type = self.model.mesh_type
            if mesh_type is not None: mesh_type = mesh_type.lower()
            grids = [m.name for m in self.document.grids.model.entries if m.name and m.type == mesh_type]
        except AttributeError:
            pass
        else:
            self.widget.mesh.clear()
            self.widget.mesh.addItems([''] + grids)
        self.widget.load_data()
        self.notify_changes = True

    def save_data_in_model(self):
        self.widget.save_data()


class FilterController(Controller):

    def __init__(self, document, model):
        super(FilterController, self).__init__(document, model)

        self.widget = QtGui.QWidget()
        layout = QtGui.QFormLayout()
        layout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)

        self.what = QtGui.QComboBox()
        self.what.addItems(PROPS)
        self.what.currentIndexChanged.connect(self.fire_changed)
        self.what.setToolTip('Name physical property to filter.')
        layout.addRow('For:', self.what)

        self.geometry = QtGui.QComboBox()
        self.geometry.setEditable(True)
        self.geometry.textChanged.connect(self.fire_changed)
        self.geometry.currentIndexChanged.connect(self.fire_changed)
        self.geometry.setCompleter(get_defines_completer(self.document.defines.model, self.widget))
        self.geometry.setToolTip('Name of the target geometry for this filter.')
        layout.addRow('Geometry:', self.geometry)

        self.widget.setLayout(layout)

    def get_widget(self):
        return self.widget

    def on_edit_enter(self):
        try:
            geometries = [g.name for g in self.document.geometry.model.roots if g.name]
        except AttributeError:
            pass
        else:
            self.geometry.clear()
            self.geometry.addItems([''] + geometries)
        self.notify_changes = False
        self.geometry.setCurrentIndex(self.geometry.findText(self.model.geometry))
        self.geometry.setEditText(self.model.geometry)
        self.what.setCurrentIndex(self.what.findText(self.model.what))
        self.notify_changes = True

    def save_data_in_model(self):
        self.model.geometry = self.geometry.currentText()
        self.model.what = self.what.currentText()


from ..model.solvers import SolversModel, CATEGORIES, SOLVERS as MODELS


class SolversController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = SolversModel()
        Controller.__init__(self, document, model)

        self._current_index = None
        self._last_index = None
        self._current_controller = None

        self.splitter = QSplitter()

        self.solvers_table = QtGui.QTableView()
        self.solvers_table.setModel(self.model)
        table_last_col_fill(self.solvers_table, self.model.columnCount(None), [100, 200])
        self.solvers_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.solvers_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.solvers_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        table_edit_shortcut(self.solvers_table, 2, 'n')
        self.splitter.addWidget(table_with_manipulators(self.solvers_table, self.splitter, title="Solvers"))

        self.parent_for_editor_widget = QtGui.QStackedWidget()
        self.splitter.addWidget(self.parent_for_editor_widget)

        self.splitter.setSizes([10000, 20000])

        focus_action = QtGui.QAction(self.solvers_table)
        focus_action.triggered.connect(lambda: self.parent_for_editor_widget.currentWidget().setFocus())
        focus_action.setShortcut(QtGui.QKeySequence(Qt.Key_Return))
        focus_action.setShortcutContext(Qt.WidgetShortcut)
        self.solvers_table.addAction(focus_action)

        selection_model = self.solvers_table.selectionModel()
        selection_model.selectionChanged.connect(self.solver_selected)

    def set_current_index(self, new_index):
        """
            Try to change current script.
            :param int new_index: index of new current script
            :return: False only when script should restore old selection
        """
        if self._current_index == new_index: return True
        if self._current_controller is not None:
            if not self._current_controller.on_edit_exit():
                return False
        self._current_index = new_index
        for i in reversed(range(self.parent_for_editor_widget.count())):
            self.parent_for_editor_widget.removeWidget(self.parent_for_editor_widget.widget(i))
        if self._current_index is None:
            self._current_controller = None
        else:
            self._current_controller = self.model.entries[new_index].get_controller(self.document)
            self.parent_for_editor_widget.addWidget(self._current_controller.get_widget())
            self._current_controller.on_edit_enter()
        return True

    def solver_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0].row() if indexes else None)):
            self.solvers_table.selectionModel().select(old_selection, QItemSelectionModel.ClearAndSelect)

    def get_widget(self):
        return self.splitter

    def save_data_in_model(self):
        if self._current_controller is not None:
            self._current_controller.save_data_in_model()

    def on_edit_enter(self):
        self.solvers_table.selectionModel().clear()   # model could have completly changed
        if self._last_index is not None:
            self.solvers_table.selectRow(self._last_index)

    def on_edit_exit(self):
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.solvers_table.selectionModel().clear()
        return True

    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.window)


class NewSolverDialog(QtGui.QDialog):

    def __init__(self, parent=None):
        super(NewSolverDialog, self).__init__(parent)
        layout = QtGui.QFormLayout()
        layout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)

        self.category = QtGui.QComboBox()
        self.category.addItems([c.title() for c in CATEGORIES])
        self.category.addItem("FILTER")
        self.category.insertSeparator(len(CATEGORIES))
        self.category.currentIndexChanged.connect(self.category_changed)
        layout.addRow("Category:", self.category)

        self.solver = QtGui.QComboBox()
        self.solver.setEditable(True)
        if MODELS:
            self.solver.setMinimumWidth(max(self.solver.fontMetrics().width(slv) for _,slv in MODELS) + 32)
            self.solver.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Preferred)
        layout.addRow("Solver:", self.solver)

        self.name = QtGui.QLineEdit()
        layout.addRow("Name:", self.name)

        button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui. QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self.setLayout(layout)

        self.category_changed(self.category.currentIndex())

    def category_changed(self, index):
        category = self.category.currentText().lower()
        self.solver.clear()
        if category == 'filter':
            self.solver.setEnabled(False)
        else:
            self.solver.setEnabled(True)
            self.solver.addItems([slv for cat,slv in MODELS if cat == category])


def get_new_solver():
    dialog = NewSolverDialog()
    if dialog.exec_() == QtGui.QDialog.Accepted:
        return dict(category=dialog.category.currentText().lower(),
                    solver=dialog.solver.currentText(),
                    name=dialog.name.text())
