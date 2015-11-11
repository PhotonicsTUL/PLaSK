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

from . import Controller, select_index_from_info
from ..utils.widgets import table_edit_shortcut, table_last_col_fill


class TableActions(object):

    def __init__(self, table, model=None):
        self.table = table
        self._model = model

    @property
    def model(self):
        return self._model if self._model is not None else self.table.model()

    def add_entry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            row = self.model.insert(index.row()+1)
        else:
            row = self.model.insert()
        if row is not None: self.table.selectRow(row)

    def remove_entry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            self.model.remove(index.row())

    def move_up(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 1 <= index < len(self.model.entries):
            self.model.swap_neighbour_entries(index-1, index)
            #self.table.selectRow(index-1)

    def move_down(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 0 <= index < len(self.model.entries)-1:
            self.model.swap_neighbour_entries(index, index+1)
            #self.table.selectRow(index+1)

    @staticmethod
    def make_action(icon, text, tip, parent, to_call, shortcut=None):
        action = QtGui.QAction(QtGui.QIcon.fromTheme(icon), text, parent)
        action.setStatusTip(tip)
        if shortcut is not None:
            action.setShortcut(shortcut)
            action.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        action.triggered.connect(to_call)
        return action

    def get(self, parent):

        self.add_action = TableActions.make_action('list-add', '&Add',
                                                    'Add new entry to the list', parent,
                                                    self.add_entry,
                                                    QtCore.Qt.CTRL + QtCore.Qt.Key_Plus)

        self.remove_action = TableActions.make_action('list-remove', '&Remove',
                                                       'Remove selected entry from the list',
                                                       parent, self.remove_entry,
                                                       QtCore.Qt.SHIFT + QtCore.Qt.Key_Delete)

        self.move_up_action = TableActions.make_action('go-up', 'Move &up',
                                                        'Change order of entries: move current entry up',
                                                        parent, self.move_up,
                                                        QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Up)

        self.move_down_action = TableActions.make_action('go-down', 'Move &down',
                                                          'Change order of entries: move current entry down',
                                                          parent, self.move_down,
                                                          QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Down)

        return self.add_action, self.remove_action, self.move_up_action, self.move_down_action


def table_and_manipulators(table, parent=None, model=None, title=None, add_undo_action=None):
    toolbar = QtGui.QToolBar()
    if model is None: model = table.model()
    toolbar.setStyleSheet("QToolBar { border: 0px }")

    if add_undo_action is None: add_undo_action = hasattr(model, 'undo_stack')
    if add_undo_action:
        toolbar.addAction(model.create_undo_action(table))
        toolbar.addAction(model.create_redo_action(table))
        toolbar.addSeparator()

    table.table_manipulators_actions = TableActions(table, model)
    actions = table.table_manipulators_actions.get(table)

    toolbar.addActions(actions)
    table.addActions(actions)

    vbox = QtGui.QVBoxLayout()
    vbox.addWidget(toolbar)
    vbox.addWidget(table)

    external = QtGui.QGroupBox()
    if title is not None:
        external.setTitle(title)
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(0)

    external.setLayout(vbox)

    return external, toolbar


def table_with_manipulators(table, parent=None, model=None, title=None, add_undo_action=None):
    return table_and_manipulators(table, parent, model, title, add_undo_action)[0]


class TableController(Controller):

    def __init__(self, document, model):
        Controller.__init__(self, document, model)
        self.table = QtGui.QTableView()
        self.table.setModel(self.model)
        self.table_actions = TableActions(self.table, self.model)

        cols = self.model.columnCount(None)  # column widths:
        for c in range(0, cols-1):
            self.table.setColumnWidth(c, 200)
            #self.table.horizontalHeader().setResizeMode(c, QtGui.QHeaderView.ResizeToContents);
        self.table.horizontalHeader().setResizeMode(cols-1, QtGui.QHeaderView.Stretch)

        self.table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        table_last_col_fill(self.table, model.columnCount(None))
        self.table.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)

        for col in range(model.columnCount()):
            label = model.headerData(col, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole)
            table_edit_shortcut(self.table, col, label[0].lower())

    def get_widget(self):
        if self.model.is_read_only():
            return self.table
        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        toolbar = QtGui.QToolBar(widget)
        toolbar.setStyleSheet("QToolBar { border: 0px }")
        toolbar.addAction(self.model.create_undo_action(widget))
        toolbar.addAction(self.model.create_redo_action(widget))
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
    #     super(TableController, self).on_edit_enter()
    #     if not self.model.is_read_only():
    #         self.document.window.set_section_actions(*self.get_table_edit_actions())

    def get_table_edit_actions(self):
        return self.table_actions.get(self.document.window)

    def on_edit_enter(self):
        self.table.setFocus()

    def select_info(self, info):
        select_index_from_info(info, self.model, self.table)



