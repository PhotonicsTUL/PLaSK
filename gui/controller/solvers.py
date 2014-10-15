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

from ..utils.widgets import table_last_col_fill
from ..utils.textedit import TextEdit
from ..utils.widgets import DEFAULT_FONT
from ..external.highlighter import SyntaxHighlighter, load_syntax
from . import Controller
from .table import table_with_manipulators
from .source import scheme, syntax


class SolverAutoWidget(QtGui.QScrollArea):

    def __init__(self, controller, parent=None):
        super(SolverAutoWidget, self).__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.controller = controller

        config = controller.model.config

        layout = QtGui.QFormLayout()
        layout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)

        label = QtGui.QLabel("General")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addRow(label)

        self.geometry = QtGui.QComboBox()
        self.geometry.setEditable(True)
        self.geometry.textChanged.connect(self.controller.fire_changed)
        self.geometry.currentIndexChanged.connect(self.controller.fire_changed)
        self.geometry.setToolTip('&lt;<b>geometry ref</b>=""&gt;<br/>'
                                 'Name of the existing geometry for use by this solver.')
        #TODO make sure the list is up-to date; add some graphical thumbnail
        layout.addRow("Geometry:", self.geometry)

        if config['mesh']:
            self.mesh = QtGui.QComboBox()
            self.mesh.setEditable(True)
            self.mesh.textChanged.connect(self.controller.fire_changed)
            self.mesh.currentIndexChanged.connect(self.controller.fire_changed)
            self.mesh.setToolTip('&lt;<b>mesh ref</b>=""&gt;<br/>'
                                 'Name of the existing {} mesh for use by this solver.'.format(config['mesh']))
            #TODO add some graphical thumbnail
            layout.addRow("Mesh:", self.mesh)
        else:
            self.mesh = None

        self.controls = {}

        for group, desc, items in config['conf']:
            label = QtGui.QLabel(desc)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            layout.addRow(label)
            if type(items) in (tuple, list):
                for item in items:
                    if len(item) == 4:
                        attr, text, help, choices = item
                        edit = QtGui.QComboBox()
                        edit.setEditable(True)
                        edit.addItems([''] + list(choices))
                        edit.textChanged.connect(self.controller.fire_changed)
                        edit.currentIndexChanged.connect(self.controller.fire_changed)
                    else:
                        attr, text, help = item
                        edit = QtGui.QLineEdit()
                        edit.textEdited.connect(self.controller.fire_changed)
                    edit.setToolTip(u'&lt;{} <b>{}</b>=""&gt;<br/>{}'.format(group, attr, help))
                    self.controls[group, attr] = edit
                    layout.addRow(text + ':', edit)
            else:
                edit = TextEdit(parent, line_numbers=False)
                font = QtGui.QFont(DEFAULT_FONT)
                font.setPointSize(font.pointSize()-1)
                edit.highlighter = SyntaxHighlighter(edit.document(), *load_syntax(syntax, scheme),
                                                     default_font=font)
                edit.setToolTip(u'&lt;<b>{0}</b>&gt;...&lt;/<b>{0}</b>&gt;<br/>{1}'.format(group, desc))
                self.controls[group] = edit
                layout.addRow(edit)
                edit.textChanged.connect(self.controller.fire_changed)

        main = QtGui.QWidget()
        main.setLayout(layout)
        self.setWidget(main)

    def resizeEvent(self, event):
        super(SolverAutoWidget, self).resizeEvent(event)
        self.widget().setFixedWidth(event.size().width())

    def eventFilter(self, obj, event):
        if obj and obj == self.widget() and event.type() == QtCore.QEvent.Resize:
            self.setMinimumWidth(obj.minimumSizeHint().width() + self.verticalScrollBar().width())
        return super(SolverAutoWidget, self).eventFilter(obj, event)

    def load_data(self):
        model = self.controller.model
        config = model.config
        self.geometry.setCurrentIndex(self.geometry.findText(model.geometry))
        self.geometry.setEditText(model.geometry)
        if self.mesh is not None:
            self.mesh.setCurrentIndex(self.mesh.findText(model.mesh))
            self.mesh.setEditText(model.mesh)
        for group, _, items in config['conf']:
            if type(items) in (tuple, list):
                for item in items:
                    attr = item[0]
                    edit = self.controls[group, attr]
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
        config = model.config
        model.geometry = self.geometry.currentText()
        if self.mesh is not None:
            model.mesh = self.mesh.currentText()
        for group, _, items in config['conf']:
            if type(items) in (tuple, list):
                for item in items:
                    attr = item[0]
                    edit = self.controls[group, attr]
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
            widget_class = self.model.config['widget']
        except KeyError:
            widget_class = SolverAutoWidget
        self.widget = widget_class(self)

    def get_widget(self):
        return self.widget

    def on_edit_enter(self):
        #TODO update geometry list
        self.notify_changes = False
        try:
            #TODO select only meshes/generators of proper dimensions
            mesh_type = self.model.config.get('mesh')
            if mesh_type is not None: mesh_type = mesh_type.lower()
            grids = [m.name for m in self.document.grids.model.entries if m.type == mesh_type]
            self.widget.mesh.clear()
            self.widget.mesh.addItems([''] + grids)
        except AttributeError:
            pass
        self.widget.load_data()
        self.notify_changes = True

    def save_data_in_model(self):
        self.widget.save_data()


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
        table_last_col_fill(self.solvers_table, self.model.columnCount(None), [80, 180])
        self.solvers_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.solvers_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.splitter.addWidget(table_with_manipulators(self.solvers_table, self.splitter, title="Solvers"))

        self.parent_for_editor_widget = QtGui.QStackedWidget()
        self.splitter.addWidget(self.parent_for_editor_widget)

        self.splitter.setSizes([10000, 20000])

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
        self.category.addItems(CATEGORIES)
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

    def category_changed(self, index):
        category = self.category.currentText()
        self.solver.clear()
        self.solver.addItems([slv for cat,slv in MODELS if cat == category])


def get_new_solver():
    dialog = NewSolverDialog()
    if dialog.exec_() == QtGui.QDialog.Accepted:
        return dict(category=dialog.category.currentText(),
                    solver=dialog.solver.currentText(),
                    name=dialog.name.text())
