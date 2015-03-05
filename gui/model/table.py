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

from . import SectionModel
from . import info


class TableModelEditMethods(object):

    def insert(self, index=None, value=None):
        if self.is_read_only(): return
        if not value:
            value = self.create_default_entry()
            if not value: return
        if 0 <= index <= len(self.entries):
            self.beginInsertRows(QtCore.QModelIndex(), index, index)
            self.entries.insert(index, value)
        else:
            index = len(self.entries)
            self.beginInsertRows(QtCore.QModelIndex(), index, index)
            self.entries.append(value)
        self.fire_changed()
        self.endInsertRows()
        return index

    def remove(self, index):
        if self.is_read_only() or index < 0 or index >= len(self.entries): return
        self.beginRemoveRows(QtCore.QModelIndex(), index, index)
        del self.entries[index]
        self.fire_changed()
        self.endRemoveRows()

    def swap_neighbour_entries(self, index1, index2):
        if self.is_read_only(): return
        if index2 < index1: index1, index2 = index2, index1
        self.beginMoveRows(QtCore.QModelIndex(), index2, index2, QtCore.QModelIndex(), index1)
        self.entries[index1], self.entries[index2] = self.entries[index2], self.entries[index1]
        self.fire_changed()
        self.endMoveRows()


class TableModel(QtCore.QAbstractTableModel, SectionModel, TableModelEditMethods):

    def __init__(self, name, parent=None, info_cb=None, *args):
        SectionModel.__init__(self, name, info_cb)
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        TableModelEditMethods.__init__(self)
        self.entries = []
        self._row_to_errors = None

    @property
    def info_by_row(self):
        """
            Allow to fast access to Info which has rows attributes and for search by row.
            :return: dict: row number -> Info
        """
        if self._row_to_errors is None:
            self._row_to_errors = {}
            for msg in self.info:
                for r in getattr(msg, 'rows', []):
                    self._row_to_errors.setdefault(r, []).append(msg)
        return self._row_to_errors

    def mark_info_invalid(self):
        self._row_to_errors = None   # this need to be refreshed
        super(TableModel, self).mark_info_invalid()

    # QAbstractTableModel implementation
    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid(): return 0
        return len(self.entries)

    def get_raw(self, col, row):
        return self.get(col, row)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return self.get(index.column(), index.row())
        if role == QtCore.Qt.ToolTipRole:
            return '\n'.join([str(err) for err in self.info_by_row.get(index.row(), []) if err.has_connection('cols', index.column())])
        if role == QtCore.Qt.DecorationRole: #QtCore.Qt.BackgroundColorRole:   #maybe TextColorRole?
            max_level = -1
            c = index.column()
            for err in self.info_by_row.get(index.row(), []):
                if err.has_connection('cols', c, c == 0):   # c == 0 -> whole row messages hav decoration only in first column
                    if err.level > max_level: max_level = err.level
            return info.infoLevelIcon(max_level)
            #c = QtGui.QPalette().color(QtGui.QPalette.Window)    #default color
            #if max_level == info.Info.ERROR: return QtGui.QColor(255, 220, 220)
            #if max_level == info.Info.WARNING: return QtGui.QColor(255, 255, 160)
            #if max_level == info.Info.INFO: return QtGui.QColor(220, 220, 255)

    def flags(self, index):
        flags = super(TableModel, self).flags(index) | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        if not self.is_read_only(): flags |= QtCore.Qt.ItemIsEditable
        #flags |= QtCore.Qt.ItemIsDragEnabled
        #flags |= QtCore.Qt.ItemIsDropEnabled

        return flags

    def set_and_fire(self, col, row, value):
        self.set(col, row, value)
        self.fire_changed()
        index = self.createIndex(row, col)
        self.dataChanged.emit(index, index)

    class SetDataCommand(QtGui.QUndoCommand):

        def __init__(self, table, col, row, new_value, QUndoCommand_parent = None):
            super(TableModel.SetDataCommand, self).__init__("change cell value to {}".format(new_value), QUndoCommand_parent)
            self.table = table
            self.col = col
            self.row = row
            self.old_value = table.get_raw(col, row)
            self.new_value = new_value

        def redo(self):
            self.table.set_and_fire(self.col, self.row, self.new_value)

        def undo(self):
            self.table.set_and_fire(self.col, self.row, self.old_value)


    def setData(self, index, value, role=QtCore.Qt.EditRole):
        #self.set(index.column(), index.row(), value)
        #self.fire_changed()
        #self.dataChanged.emit(index, index)
        self.undo_stack.push(TableModel.SetDataCommand(self, index.column(), index.row(), value))
        return True
