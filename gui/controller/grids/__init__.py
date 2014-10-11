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
from ...qt.QtGui import QSplitter, QItemSelectionModel

from .. import Controller
from ...utils.widgets import table_last_col_fill
from ..table import table_with_manipulators
from ...model.grids import GridsModel


class GridsController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = GridsModel()
        Controller.__init__(self, document, model)

        self.current_index = None
        self.current_controller = None

        self.splitter = QSplitter()

        self.grids_table = QtGui.QTableView()
        self.grids_table.setModel(self.model)
        #self.grids_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model, self.grids_table))
        #self.materialsTableActions = TableActions(self.grids_table)
        table_last_col_fill(self.grids_table, self.model.columnCount(None), 80)
        self.grids_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.grids_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.splitter.addWidget(table_with_manipulators(self.grids_table, self.splitter, title="Meshes and generators"))

        self.parent_for_editor_widget = QtGui.QStackedWidget()
        self.splitter.addWidget(self.parent_for_editor_widget)

        self.splitter.setSizes([10000,26000])

        selection_model = self.grids_table.selectionModel()
        selection_model.selectionChanged.connect(self.grid_selected) #currentChanged ??

    def set_current_index(self, new_index):
        """
            Try to change current script.
            :param int new_index: index of new current script
            :return: False only when script should restore old selection
        """
        if self.current_index == new_index: return True
        if self.current_controller is not None:
            if not self.current_controller.on_edit_exit():
                return False
        self.current_index = new_index
        for i in reversed(range(self.parent_for_editor_widget.count())):
            self.parent_for_editor_widget.removeWidget(self.parent_for_editor_widget.widget(i))
        if self.current_index is None:
            self.current_controller = None
        else:
            self.current_controller = self.model.entries[new_index].get_controller(self.document)
            self.parent_for_editor_widget.addWidget(self.current_controller.get_widget())
            self.current_controller.on_edit_enter()
        return True

    def grid_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0].row() if indexes else None)):
            self.grids_table.selectionModel().select(old_selection, QItemSelectionModel.ClearAndSelect)

    def get_widget(self):
        return self.splitter

    def save_data_in_model(self):
        if self.current_controller is not None:
            self.current_controller.save_data_in_model()

    def on_edit_enter(self):
        #if self.current_controller is not None:
        #    self.current_controller.on_edit_enter()
        self.grids_table.selectionModel().clear()   #model could completly changed

    def on_edit_exit(self):
        if self.current_controller is not None:
            self.grids_table.selectionModel().clear()
        return True

    #def onEditEnter(self):
    #    self.saveDataInModel()  #this should do nothing, but is called in case of subclass use it
    #    if not self.model.isReadOnly():
    #        self.document.window.setSectionActions(*self.get_table_edit_actions())

    # when editor is turn off, model should be update
    #def onEditExit(self):
    #    self.document.window.setSectionActions()

    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.window)
