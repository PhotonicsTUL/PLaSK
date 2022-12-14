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


from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from . import Controller, select_index_from_info
from ..utils.widgets import table_edit_shortcut, table_last_col_fill, create_undo_actions, set_icon_size
from ..utils.config import CONFIG


def top_level_index(index):
    """Find top level index (child of the root) by traversing the tree upwards."""
    prev = index
    while index.isValid():
        prev = index
        index = index.parent()
    return prev


class TableActions:

    def __init__(self, table, model=None):
        self.table = table
        self._model = model

    @property
    def model(self):
        return self._model if self._model is not None else self.table.model()

    def top_level_selection_index(self):
        return top_level_index(self.table.selectionModel().currentIndex())

    def select_row(self, row):
        if row is None: return
        if isinstance(self.table, QTableView):
            self.table.selectRow(row)
        else:
            self.table.selectionModel().setCurrentIndex(self.model.index(row, 0, QModelIndex()),
                                               QItemSelectionModel.SelectionFlag.SelectCurrent | QItemSelectionModel.SelectionFlag.Rows | QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def add_entry(self):
        index = self.top_level_selection_index()
        if index.isValid():
            row = self.model.insert(index.row()+1)
        else:
            row = self.model.insert()
        self.select_row(row)

    def remove_entry(self):
        index = self.top_level_selection_index()
        if index.isValid():
            self.model.remove(index.row())

    def move_up(self):
        index = self.top_level_selection_index()
        if not index.isValid(): return
        index = index.row()
        if 1 <= index < len(self.model.entries):
            self.model.swap_entries(index-1, index)
            #self.table.selectRow(index-1)

    def move_down(self):
        index = self.top_level_selection_index()
        if not index.isValid(): return
        index = index.row()
        if 0 <= index < len(self.model.entries)-1:
            self.model.swap_entries(index, index+1)
            #self.table.selectRow(index+1)

    @staticmethod
    def make_action(icon, text, tip, parent, to_call, shortcut=None):
        action = QAction(QIcon.fromTheme(icon), text, parent)
        action.setStatusTip(tip)
        if shortcut is not None:
            CONFIG.set_shortcut(action, shortcut)
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        action.triggered.connect(to_call)
        return action

    def get(self, parent):
        self.add_action = TableActions.make_action('list-add', '&Add Entry',
                                                    'Add new entry to the list', parent,
                                                    self.add_entry,
                                                    'entry_add')

        self.remove_action = TableActions.make_action('list-remove', '&Remove Entry',
                                                       'Remove selected entry from the list',
                                                       parent, self.remove_entry,
                                                       'entry_remove')

        self.move_up_action = TableActions.make_action('go-up', 'Move &Up',
                                                        'Change order of entries: move current entry up',
                                                        parent, self.move_up,
                                                        'entry_move_up')

        self.move_down_action = TableActions.make_action('go-down', 'Move &Down',
                                                          'Change order of entries: move current entry down',
                                                          parent, self.move_down,
                                                          'entry_move_down')

        return self.add_action, self.remove_action, self.move_up_action, self.move_down_action


def table_and_manipulators(table, parent=None, model=None, title=None, add_undo_action=None):
    toolbar = QToolBar()
    if model is None: model = table.model()
    toolbar.setStyleSheet("QToolBar { border: 0px }")
    set_icon_size(toolbar)

    if add_undo_action is None: add_undo_action = hasattr(model, 'undo_stack')
    if add_undo_action:
        create_undo_actions(toolbar, model, table)
        toolbar.addSeparator()

    table.table_manipulators_actions = TableActions(table, model)
    actions = table.table_manipulators_actions.get(table)

    toolbar.addActions(actions)
    table.addActions(actions)

    vbox = QVBoxLayout()
    vbox.addWidget(toolbar)
    vbox.addWidget(table)

    if title is not None:
        external = QGroupBox()
        external.setTitle(title)
    else:
        external = QWidget()
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(0)

    external.setLayout(vbox)

    return external, toolbar


def table_with_manipulators(table, parent=None, model=None, title=None, add_undo_action=None):
    return table_and_manipulators(table, parent, model, title, add_undo_action)[0]


class TableController(Controller):

    def __init__(self, document, model):
        Controller.__init__(self, document, model)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table_actions = TableActions(self.table, self.model)

        cols = self.model.columnCount(None)  # column widths:
        for c in range(0, cols-1):
            self.table.setColumnWidth(c, 200)
            #self.table.horizontalHeader().setResizeMode(c, QHeaderView.ResizeMode.ResizeToContents);
        try:
            self.table.horizontalHeader().setResizeMode(cols-1, QHeaderView.ResizeMode.Stretch)
        except AttributeError:
            self.table.horizontalHeader().setSectionResizeMode(cols-1, QHeaderView.ResizeMode.Stretch)

        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table_last_col_fill(self.table, model.columnCount(None))
        try:
            self.table.horizontalHeader().setResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        except AttributeError:
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        for col in range(model.columnCount()):
            label = model.headerData(col, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            table_edit_shortcut(self.table, col, label[0].lower())

    def get_widget(self):
        if self.model.is_read_only():
            return self.table
        widget = QWidget()
        layout = QVBoxLayout()
        toolbar = QToolBar(widget)
        toolbar.setStyleSheet("QToolBar { border: 0px }")
        set_icon_size(toolbar)
        create_undo_actions(toolbar, self.model, widget)
        toolbar.addSeparator()
        actions = self.get_table_edit_actions()
        for a in actions:
            if not a:
                toolbar.addSeparator()
            else:
                toolbar.addAction(a)
        self.table.addActions(actions)
        layout.addWidget(toolbar)
        layout.addWidget(self.table)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        widget.setLayout(layout)
        return widget

    # def on_edit_enter(self):
    #     super().on_edit_enter()
    #     if not self.model.is_read_only():
    #         self.document.window.set_section_actions(*self.get_table_edit_actions())

    def get_table_edit_actions(self):
        return self.table_actions.get(self.document.window)

    def on_edit_enter(self):
        self.table.setFocus()

    def select_info(self, info):
        select_index_from_info(info, self.model, self.table)
