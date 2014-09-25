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

from ..qt import QtGui, QtCore

from ..model.defines import DefinesModel
from .table import TableController


class AfterBracketCompleter(QtGui.QCompleter):

    def pathFromIndex(self, index):
        path = QtGui.QCompleter.pathFromIndex(self, index)

        try:
            text = self.widget().text()         # text field
        except AttributeError:
            text = self.widget().currentText()  # combo box

        lst = text.rsplit('{', 1)
        if len(lst) > 1:
            path = '%s{%s}' % (lst[0], path)
        else:
            path = '{%s}' % path

        return path

    def splitPath(self, path):
        path = path.rsplit('{', 1)[-1].lstrip(' ')
        return [path]


class DefineHintsTableModel(QtCore.QAbstractTableModel):

    def __init__(self, defines_model, parent=None, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)   #QtCore.QObject.parent(defines_model)
        self.model = defines_model

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.model.rowCount(parent)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid() and index.column() == 1:
            if role == QtCore.Qt.FontRole:
                font = QtGui.QFont()
                font.setItalic(True)
                return font
            if role == QtCore.Qt.TextColorRole:
                return QtGui.QColor(90, 90, 90) #QtGui.QColor(QtCore.Qt.blue)
        return self.model.data(index, role)

    #def flags(self, index):
    #    return super(DefineHintsTableModel, self).flags(index) | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2

    def headerData(self, col, orientation, role):
        return self.model.headerData(col, orientation, role)


class DefinesCompletionDelegate(QtGui.QStyledItemDelegate):

    def __init__(self, model, parent):
        QtGui.QStyledItemDelegate.__init__(self, parent)
        self.model = DefineHintsTableModel(model, parent)
        #self.model = model

    def get_defines_completer(self, parent):
        completer = AfterBracketCompleter(self.model, parent)
        completer.setModel(self.model)  # PySide needs this
        tab = QtGui.QTableView(parent)
        #tab.resizeColumnsToContents()
        tab.setModel(self.model)
        tab.setMinimumSize(0, 200)
        #tab.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        tab.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        tab.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        tab.setSelectionBehavior(QtGui.QTableView.SelectRows)
        tab.setSelectionMode(QtGui.QTableView.SingleSelection)
        tab.horizontalHeader().hide()
        tab.verticalHeader().hide()
        tab.setSortingEnabled(False)
        tab.setShowGrid(False)
        tab.setWordWrap(False)
        #tab.setContentsMargins(1, 1, 1, 1)
        completer.setPopup(tab)
        return completer

    def createEditor(self, parent, option, index):
        ed = QtGui.QLineEdit(parent)
        completer = self.get_defines_completer(parent)
        ed.setCompleter(completer)
        return ed

    #def setEditorData(self, editor, index):
    #    editor.blockSignals(True)
    #    editor.setCurrentIndex(int(index.model().data(index)))
    #    editor.blockSignals(False)

    #def setModelData(self, editor, model, index):
    #    model.setData(index, editor.currentIndex())

    #@QtCore.pyqtSlot()
    #def currentIndexChanged(self):
    #    self.commitData.emit(self.sender())


class DefinesController(TableController):

    def __init__(self, document, model=None):
        if model is None: model = DefinesModel()
        TableController.__init__(self, document, model)
        self.table.setItemDelegateForColumn(1, DefinesCompletionDelegate(self.model, self.table))
