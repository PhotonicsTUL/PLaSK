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

from ..qt.QtGui import *
from ..qt.QtWidgets import *
from ..utils.qsignals import BlockQtSignals

from ..model.defines import DefinesModel
from .table import TableController


class AfterBracketCompleter(QCompleter):

    def pathFromIndex(self, index):
        path = QCompleter.pathFromIndex(self, index)

        try:
            text = self.widget().text()         # text field
        except AttributeError:
            text = self.widget().currentText()  # combo box

        lst = text.rsplit('{', 1)
        if len(lst) > 1:
            path = '%s%s' % (lst[0], path)

        return path

    def splitPath(self, path):
        path = path.rsplit('{', 1)
        if len(path) > 1:
            return ['{'+path[-1].lstrip(' ')]
        elif isinstance(path, (list, tuple)):
            return path
        else:
            return [path]


class DefinesCompleterModel(QAbstractTableModel):

    def __init__(self, model, parent=None, strings=None):
        super().__init__(parent)
        self.strings = [] if strings is None else strings
        self.defines = model

    def make_file_xml_element(self):
        return self.defines.make_file_xml_element()

    def data(self, index, role=Qt.ItemDataRole.EditRole):
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            ls = len(self.strings)
            if index.row() < ls:
                if col == 0:
                    return self.strings[index.row()]
                else:
                    return None
            else:
                if col == 0:
                    return '{{{}}}'.format(self.defines.get(0, index.row()-ls))
                else:
                    return self.defines.get(col, index.row()-ls)
        elif role == Qt.ItemDataRole.ForegroundRole and col != 0:
            return QColor(Qt.GlobalColor.gray)

    def columnCount(self, parent=QModelIndex()):
        return 2

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid(): return 0
        return len(self.strings) + len(self.defines.entries)

    def headerData(self, col, orientation, role):
        return self.defines.headerData(col, orientation, role)


def get_defines_completer(model, parent, strings=None):
    if isinstance(model, QCompleter): model = model.model()
    if isinstance(model, DefinesCompleterModel): model = model.defines
    if strings is None and isinstance(parent, QComboBox):
        strings = [parent.itemText(i) for i in range(parent.count())]
    model = DefinesCompleterModel(model, parent, strings)
    completer = AfterBracketCompleter(model, parent)
    completer.setModel(model)  # PySide needs this
    tab = QTableView(parent)
    #tab.resizeColumnsToContents()
    tab.setModel(model)
    tab.setMinimumSize(0, 200)
    #tab.horizontalHeader().setResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    try:
        tab.horizontalHeader().setResizeMode(QHeaderView.ResizeMode.Stretch)
    except AttributeError:
        tab.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    tab.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    tab.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    tab.setSelectionMode(QTableView.SelectionMode.SingleSelection)
    tab.horizontalHeader().hide()
    tab.verticalHeader().hide()
    tab.setSortingEnabled(False)
    tab.setShowGrid(False)
    tab.setWordWrap(False)
    #tab.setContentsMargins(1, 1, 1, 1)
    completer.setPopup(tab)
    return completer


class DefinesCompletionDelegate(QStyledItemDelegate):

    def __init__(self, model, parent):
        QStyledItemDelegate.__init__(self, parent)
        self.model = DefinesCompleterModel(model, parent)
        #self.model = model

    def createEditor(self, parent, option, index):
        ed = QLineEdit(parent)
        completer = get_defines_completer(self.model, parent)
        ed.setCompleter(completer)
        return ed

    def setEditorData(self, editor, index):
        with BlockQtSignals(editor):
            super().setEditorData(editor, index)

    # def setEditorData(self, editor, index):
    #     with BlockQtSignals(editor):
    #         editor.setCurrentIndex(int(index.model().data(index)))

    #def setModelData(self, editor, model, index):
    #    model.setData(index, editor.currentIndex())

    #@pyqtSlot()
    #def currentIndexChanged(self):
    #    self.commitData.emit(self.sender())


class DefinesController(TableController):

    def __init__(self, document, model=None):
        if model is None: model = DefinesModel()
        TableController.__init__(self, document, model)
        self.table.setItemDelegateForColumn(1, DefinesCompletionDelegate(self.model, self.table))
