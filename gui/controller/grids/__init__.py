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
from ...qt.QtCore import Qt
from ...qt.QtGui import QSplitter, QItemSelectionModel

from .. import Controller, select_index_from_info
from ...utils.widgets import table_last_col_fill, table_edit_shortcut
from ..table import table_with_manipulators
from ...model.grids import GridsModel

# TODO use ControllerWithSubController (?)
class GridsController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = GridsModel()
        Controller.__init__(self, document, model)

        self._current_index = None
        self._last_index = None
        self._current_controller = None

        self.splitter = QSplitter()

        self.grids_table = QtGui.QTableView()
        self.grids_table.setModel(self.model)
        #self.grids_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model, self.grids_table))
        #self.materialsTableActions = TableActions(self.grids_table)
        table_last_col_fill(self.grids_table, self.model.columnCount(None), 80)
        self.grids_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.grids_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        table_edit_shortcut(self.grids_table, 0, 'n')
        self.splitter.addWidget(table_with_manipulators(self.grids_table, self.splitter, title="Meshes and Generators"))
        self.grids_table.setVisible(False)
        self.grids_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.grids_table.setVisible(True)

        self.parent_for_editor_widget = QtGui.QStackedWidget()
        self.splitter.addWidget(self.parent_for_editor_widget)

        focus_action = QtGui.QAction(self.grids_table)
        focus_action.triggered.connect(lambda: self.parent_for_editor_widget.currentWidget().setFocus())
        focus_action.setShortcut(QtGui.QKeySequence(Qt.Key_Return))
        focus_action.setShortcutContext(Qt.WidgetShortcut)
        self.grids_table.addAction(focus_action)

        self.splitter.setSizes([10000,26000])

        selection_model = self.grids_table.selectionModel()
        selection_model.selectionChanged.connect(self.grid_selected) #currentChanged ??

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

    def grid_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0].row() if indexes else None)):
            self.grids_table.selectionModel().select(old_selection, QtGui.QItemSelectionModel.ClearAndSelect)

    def get_widget(self):
        return self.splitter

    def save_data_in_model(self):
        if self._current_controller is not None:
            self._current_controller.save_data_in_model()

    def on_edit_enter(self):
        self.grids_table.selectionModel().clear()   # model could completly changed
        if self._last_index is not None:
            self.grids_table.selectRow(self._last_index)
        self.grids_table.setFocus()

    def on_edit_exit(self):
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.grids_table.selectionModel().clear()
        return True

    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.window)

    def select_info(self, info):
        if select_index_from_info(info, self.model, self.grids_table):
            #TODO try to select property
            pass