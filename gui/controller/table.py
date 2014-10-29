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

from . import Controller

class TableActions(object):

    def __init__(self, table, model=None):
        self.table = table
        self.model = model if model is not None else table.model()

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

    def get(self, parent):
        self.add_action = QtGui.QAction(QtGui.QIcon.fromTheme('list-add'),
                                        '&Add', parent)
        self.add_action.setStatusTip('Add new entry to the list')
        self.add_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Plus)
        self.add_action.triggered.connect(self.add_entry)

        self.remove_action = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove'),
                                           '&Remove', parent)
        self.remove_action.setStatusTip('Remove selected entry from the list')
        # self.remove_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Minus)
        self.remove_action.triggered.connect(self.remove_entry)

        self.move_up_action = QtGui.QAction(QtGui.QIcon.fromTheme('go-up'),
                                            'Move &up', parent)
        self.move_up_action.setStatusTip('Change order of entries: move current entry up')
        self.move_up_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Up)
        self.move_up_action.triggered.connect(self.move_up)

        self.move_down_action = QtGui.QAction(QtGui.QIcon.fromTheme('go-down'),
                                              'Move &down', parent)
        self.move_down_action.setStatusTip('Change order of entries: move current entry down')
        self.move_down_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Down)
        self.move_down_action.triggered.connect(self.move_down)

        return self.add_action, self.remove_action, self.move_up_action, self.move_down_action


def table_and_manipulators(table, parent=None, model=None, title=None):
    toolbar = QtGui.QToolBar()
    toolbar.setStyleSheet("QToolBar { border: 0px }")
    table.table_manipulators_actions = TableActions(table, model)
    toolbar.addActions(table.table_manipulators_actions.get(parent))

    vbox = QtGui.QVBoxLayout()
    vbox.addWidget(toolbar)
    vbox.addWidget(table)

    external = QtGui.QGroupBox()
    if title is not None:
        external.setTitle(title)
        m = external.getContentsMargins()
        external.setContentsMargins(0, m[1], 0, m[3])
    else:
        external.setContentsMargins(0, 0, 0, 0)
    vbox.setContentsMargins(0, 0, 0, 0)

    external.setLayout(vbox)

    return external, toolbar


def table_with_manipulators(table, parent=None, model=None, title=None):
    return table_and_manipulators(table, parent, model, title)[0]


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

    def get_widget(self):
        if self.model.is_read_only():
            return self.table
        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        toolbar = QtGui.QToolBar(widget)
        toolbar.setStyleSheet("QToolBar { border: 0px }")
        for a in self.get_table_edit_actions():
            if not a:
                toolbar.addSeparator()
            else:
                toolbar.addAction(a)
        layout.addWidget(toolbar)
        layout.addWidget(self.table)
        widget.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        return widget

    # def on_edit_enter(self):
    #     super(TableController, self).on_edit_enter()
    #     if not self.model.is_read_only():
    #         self.document.window.set_section_actions(*self.get_table_edit_actions())

    def get_table_edit_actions(self):
        return self.table_actions.get(self.document.window)
