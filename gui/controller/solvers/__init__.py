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
from itertools import groupby

from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from ...model.connects import PROPS
from ...utils.widgets import table_last_col_fill, table_edit_shortcut
from .. import Controller, select_index_from_info
from ..table import table_with_manipulators
from ..defines import get_defines_completer
from .autosolver import SolverAutoWidget


def _solvers_key(slv):
    try:
        i = SUFFIXES.index(suffix(slv))
    except ValueError:
        i = len(SUFFIXES)
    return i, slv


class FilterController(Controller):

    def __init__(self, document, model):
        super(FilterController, self).__init__(document, model)

        self.widget = QWidget()
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.what = QComboBox()
        self.what.addItems(PROPS)
        self.what.currentIndexChanged.connect(self.fire_changed)
        self.what.setToolTip('Name physical property to filter.')
        layout.addRow('For:', self.what)

        self.geometry = QComboBox()
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
        with self.mute_changes():
            self.geometry.setCurrentIndex(self.geometry.findText(self.model.geometry))
            self.geometry.setEditText(self.model.geometry)
            self.what.setCurrentIndex(self.what.findText(self.model.what))

    def save_data_in_model(self):
        self.model.geometry = self.geometry.currentText()
        self.model.what = self.what.currentText()


from ...model.solvers import SolversModel, CATEGORIES, SOLVERS as MODELS, suffix, SUFFIXES, update_solvers


class SolversController(Controller):

    """
        :param document:
        :param SolversModel model:
    """
    def __init__(self, document, model=None):
        if model is None: model = SolversModel()
        Controller.__init__(self, document, model)

        self._current_index = None
        self._last_index = None
        self._current_controller = None

        self.splitter = QSplitter()

        self.solvers_table = QTableView()
        self.solvers_table.setModel(self.model)
        table_last_col_fill(self.solvers_table, self.model.columnCount(None), [100, 200])
        self.solvers_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.solvers_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        try:
            self.solvers_table.horizontalHeader().setResizeMode(QHeaderView.ResizeToContents)
        except AttributeError:
            self.solvers_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table_edit_shortcut(self.solvers_table, 2, 'n')
        self.splitter.addWidget(table_with_manipulators(self.solvers_table, self.splitter, title="Solvers"))

        self.parent_for_editor_widget = QStackedWidget()
        self.splitter.addWidget(self.parent_for_editor_widget)

        self.splitter.setSizes([10000, 20000])

        focus_action = QAction(self.solvers_table)
        focus_action.triggered.connect(lambda: self.parent_for_editor_widget.currentWidget().setFocus())
        focus_action.setShortcut(QKeySequence(Qt.Key_Return))
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
        update_solvers(self.document.filename, self)
        self.solvers_table.selectionModel().clear()   # model could have completely changed
        if self._last_index is not None:
            self.solvers_table.selectRow(self._last_index)

    def on_edit_exit(self):
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.solvers_table.selectionModel().clear()
        return True

    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.window)

    def select_info(self, info):
        if select_index_from_info(info, self.model, self.solvers_table):
            #TODO try to select property of solver
            pass


class NewSolverDialog(QDialog):

    def __init__(self, model, parent=None):

        super(NewSolverDialog, self).__init__(parent)
        self.setWindowTitle('Create New Solver')
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.model = model

        self.category = QComboBox()
        categories = CATEGORIES + model.local_categories
        categories += ([] if categories[-1] is None else [None])
        self.category.addItems([c.title() for c in categories if c is not None])
        self.category.addItem("FILTER")
        seps = (n for n,c in enumerate(categories) if c is None)
        for sep in seps:
            self.category.insertSeparator(sep)
        self.category.setCurrentIndex(5)
        self.category.currentIndexChanged.connect(self.category_changed)
        layout.addRow("C&ategory:", self.category)

        self.solver = QComboBox()
        self.solver.setEditable(True)
        if MODELS:
            self.solver.setMinimumWidth(max(self.solver.fontMetrics().width(slv) for _,slv in MODELS) + 32)
            self.solver.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        layout.addRow("&Solver:", self.solver)

        self.name = QLineEdit()
        layout.addRow("&Name:", self.name)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok |  QDialogButtonBox.Cancel)
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
            models = MODELS
            models.update(self.model.local_solvers)
            solvers = [slv for cat, slv in models if cat == category]
            solvers.sort(key=_solvers_key)
            self.solver.addItems(solvers)
            grps = [len(list(g)) for _, g in groupby(solvers, suffix)]
            i = -1
            for l in grps[:-1]:
                i += l + 1
                self.solver.insertSeparator(i)


def get_new_solver(model):
    dialog = NewSolverDialog(model)
    if dialog.exec_() == QDialog.Accepted:
        return dict(category=dialog.category.currentText().lower(),
                    solver=dialog.solver.currentText(),
                    name=dialog.name.text())
