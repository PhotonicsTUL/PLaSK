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

from ..qt.QtCore import *

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from . import SectionModel
from . import info


class TableModelEditMethods(object):

    def _exec_command(self, command):
        if hasattr(self, 'undo_stack'):
            self.undo_stack.push(command)
        else:
            command.redo()

    class InsertEntryCommand(QUndoCommand):

        def __init__(self, table, index, entry, QUndoCommand_parent = None):
            super(TableModelEditMethods.InsertEntryCommand, self).__init__('insert row {}'.format(index+1), QUndoCommand_parent)
            self.index = index
            self.table = table
            self.entry = entry

        def redo(self):
            self.table.beginInsertRows(QModelIndex(), self.index, self.index)
            self.table.entries.insert(self.index, self.entry)
            self.table.fire_changed()
            self.table.endInsertRows()

        def undo(self):
            self.table.beginRemoveRows(QModelIndex(), self.index, self.index)
            del self.table.entries[self.index]
            self.table.fire_changed()
            self.table.endRemoveRows()

    def insert(self, index=None, value=None):
        if self.is_read_only(): return
        if value is None:
            value = self.create_default_entry()
            if value is None: return
        if index is None or index < 0: index = 0
        if index >= len(self.entries): index = len(self.entries)
        self._exec_command(TableModelEditMethods.InsertEntryCommand(self, index, value))
        return index


    class RemoveEntryCommand(QUndoCommand):

        def __init__(self, table, index, QUndoCommand_parent = None):
            super(TableModelEditMethods.RemoveEntryCommand, self).__init__('remove row {}'.format(index+1), QUndoCommand_parent)
            self.index = index
            self.table = table
            self.removed_entry = self.table.entries[self.index]

        def redo(self):
            self.table.beginRemoveRows(QModelIndex(), self.index, self.index)
            del self.table.entries[self.index]
            self.table.fire_changed()
            self.table.endRemoveRows()

        def undo(self):
            self.table.beginInsertRows(QModelIndex(), self.index, self.index)
            self.table.entries.insert(self.index, self.removed_entry)
            self.table.fire_changed()
            self.table.endInsertRows()


    def remove(self, index):
        if self.is_read_only() or index < 0 or index >= len(self.entries): return
        self._exec_command(TableModelEditMethods.RemoveEntryCommand(self, index))


    class SwapEntriesCommand(QUndoCommand):

        def __init__(self, table, index1, index2, QUndoCommand_parent = None):
            if index2 < index1:
                self.index1, self.index2 = index2, index1
            else:
                self.index1, self.index2 = index1, index2
            super(TableModelEditMethods.SwapEntriesCommand, self).__init__('swap entries at rows {} and {}'.format(index1+1, index2+1), QUndoCommand_parent)
            self.table = table

        def redo(self):
            self.table.beginMoveRows(QModelIndex(), self.index2, self.index2, QModelIndex(), self.index1)
            self.table.entries[self.index1], self.table.entries[self.index2] =\
                        self.table.entries[self.index2], self.table.entries[self.index1]
            self.table.fire_changed()
            self.table.endMoveRows()

        def undo(self):
            self.redo()

    def swap_entries(self, index1, index2):
        if self.is_read_only(): return
        self._exec_command(TableModelEditMethods.SwapEntriesCommand(self, index1, index2))


    def set_and_fire(self, col, row, value):
        self.set(col, row, value)
        self.fire_changed()
        index = self.createIndex(row, col)
        self.dataChanged.emit(index, index)

    class SetDataCommand(QUndoCommand):

        def __init__(self, table, col, row, new_value, QUndoCommand_parent = None):
            super(TableModel.SetDataCommand, self)\
                .__init__(u'change cell value at row {} to "{}"'.format(row+1, new_value), QUndoCommand_parent)
            self.table = table
            self.col = col
            self.row = row
            self.old_value = table.get_raw(col, row)
            self.new_value = new_value

        def redo(self):
            self.table.set_and_fire(self.col, self.row, self.new_value)

        def undo(self):
            self.table.set_and_fire(self.col, self.row, self.old_value)

    def setData(self, index, value, role=Qt.EditRole):
        #self.set(index.column(), index.row(), value)
        #self.fire_changed()
        #self.dataChanged.emit(index, index)
        if self.is_read_only() or not index.isValid() or v6alue == self.data(index):
            return False
        self._exec_command(TableModel.SetDataCommand(self, index.column(), index.row(), value))
        return True

    class SetEntriesCommand(QUndoCommand):

        def __init__(self, table, new_entries, QUndoCommand_parent = None):
            super(TableModel.SetEntriesCommand, self).__init__('edit XPL source', QUndoCommand_parent)
            self.table = table
            self.old_entries = table.entries
            self.new_entries = new_entries

        def _set_entries(self, entries):
            self.table.beginResetModel()
            self.table.entries = entries
            self.table.endResetModel()
            self.table.fire_changed()

        def redo(self):
            self._set_entries(self.new_entries)

        def undo(self):
            self._set_entries(self.old_entries)

    def _set_entries(self, new_entries, undoable = True):
        command = TableModel.SetEntriesCommand(self, new_entries)
        if undoable:
            self._exec_command(command)
        else:
            command.redo()
            if hasattr(self, 'undo_stack'): self.undo_stack.clear()


class TableModel(TableModelEditMethods, SectionModel, QAbstractTableModel):

    def __init__(self, name, parent=None, info_cb=None, *args):
        SectionModel.__init__(self, name, info_cb)
        QAbstractTableModel.__init__(self, parent)
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
            for msg in self.get_info():
                for r in getattr(msg, 'rows', []):
                    self._row_to_errors.setdefault(r, []).append(msg)
        return self._row_to_errors

    def mark_info_invalid(self):
        self._row_to_errors = None   # this need to be refreshed
        super(TableModel, self).mark_info_invalid()

    # QAbstractTableModel implementation
    def rowCount(self, parent=QModelIndex()):
        if parent.isValid(): return 0
        return len(self.entries)

    def get_raw(self, col, row):
        return self.get(col, row)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return None
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return self.get(index.column(), index.row())
        if role == Qt.ToolTipRole:
            return '\n'.join([str(err) for err in self.info_by_row.get(index.row(), []) if err.has_connection('cols', index.column())])
        if role == Qt.DecorationRole: #Qt.BackgroundColorRole:   #maybe TextColorRole?
            max_level = -1
            c = index.column()
            for err in self.info_by_row.get(index.row(), []):
                if err.has_connection('cols', c, c == 0):   # c == 0 -> whole row messages hav decoration only in first column
                    if err.level > max_level: max_level = err.level
            return info.info_level_icon(max_level)
            #c = QPalette().color(QPalette.Window)    #default color
            #if max_level == info.Info.ERROR: return QColor(255, 220, 220)
            #if max_level == info.Info.WARNING: return QColor(255, 255, 160)
            #if max_level == info.Info.INFO: return QColor(220, 220, 255)

    def flags(self, index):
        flags = super(TableModel, self).flags(index) | Qt.ItemIsSelectable | Qt.ItemIsEnabled

        if not self.is_read_only(): flags |= Qt.ItemIsEditable
        #flags |= Qt.ItemIsDragEnabled
        #flags |= Qt.ItemIsDropEnabled

        return flags
