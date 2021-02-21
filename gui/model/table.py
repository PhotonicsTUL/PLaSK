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


class TableModelEditMethods:

    def _exec_command(self, command):
        if hasattr(self, 'undo_stack'):
            self.undo_stack.push(command)
        else:
            command.redo()

    class InsertEntryCommand(QUndoCommand):

        def __init__(self, table, index, entry, parent=None):
            super(TableModelEditMethods.InsertEntryCommand, self).\
                __init__('insert row {} to {}'.format(index+1, table.name), parent)
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

        def __init__(self, table, index, parent=None):
            super(TableModelEditMethods.RemoveEntryCommand, self).\
                __init__('remove row {} from {}'.format(index+1, table.name), parent)
            self.index = index
            self.table = table
            self.removed_entry = self.table.entries[self.index]

        def redo(self):
            self.table.beginRemoveRows(QModelIndex(), self.index, self.index)
            del self.table.entries[self.index]
            self.table.fire_changed()
            self.table.endRemoveRows()
            if len(self.removed_entry.comments) > 0:
                if self.index < len(self.table.entries):
                    self.table.entries[self.index].comments = self.removed_entry.comments + \
                                                              self.table.entries[self.index].comments
                else:
                    self.table.endcomments = self.removed_entry.comments + self.table.endcomments

        def undo(self):
            strip_comments = len(self.removed_entry.comments)
            if strip_comments > 0:
                if self.index < len(self.table.entries):
                    self.table.entries[self.index].comments = self.table.entries[self.index].comments[strip_comments:]
                else:
                    self.table.endcomments = self.table.endcomments[strip_comments:]
            self.table.beginInsertRows(QModelIndex(), self.index, self.index)
            self.table.entries.insert(self.index, self.removed_entry)
            self.table.fire_changed()
            self.table.endInsertRows()

    def remove(self, index):
        if self.is_read_only() or index < 0 or index >= len(self.entries): return
        self._exec_command(TableModelEditMethods.RemoveEntryCommand(self, index))

    class SwapEntriesCommand(QUndoCommand):

        def __init__(self, table, index1, index2, parent=None):
            if index2 < index1:
                self.index1, self.index2 = index2, index1
            else:
                self.index1, self.index2 = index1, index2
            super(TableModelEditMethods.SwapEntriesCommand, self).\
                __init__('swap entries at rows {} and {} in {}'.format(index1+1, index2+1, table.name), parent)
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

        def __init__(self, table, col, row, new_value, parent=None, merge_id=-1):
            self.table = table
            self.col = col
            self.row = row
            self.old_value = table.get_raw(col, row)
            self.new_value = new_value
            self._id = merge_id
            super(TableModel.SetDataCommand, self).__init__(self._get_title(), parent)

        def _get_title(self):
            col_name = self.table.headerData(self.col, Qt.Horizontal, Qt.DisplayRole).lower()
            return u"change {} at row {} to '{}' in {}".format(col_name, self.row+1, self.new_value, self.table.name)

        def id(self):
            return self._id

        def redo(self):
            self.table.set_and_fire(self.col, self.row, self.new_value)

        def undo(self):
            self.table.set_and_fire(self.col, self.row, self.old_value)

        def mergeWith(self, other):
            if self.table is not other.table or self.row != other.row or self.col != other.col:
                return False
            self.new_value = other.new_value
            self.setText(self._get_title())
            return True

    def setData(self, index, value, role=Qt.EditRole, merge_id=-1):
        #self.set(index.column(), index.row(), value)
        #self.fire_changed()
        #self.dataChanged.emit(index, index)
        if self.is_read_only() or not index.isValid() or value == self.data(index):
            return False
        self._exec_command(TableModel.SetDataCommand(self, index.column(), index.row(), value, merge_id=merge_id))
        return True

    class SetEntriesCommand(QUndoCommand):

        def __init__(self, table, new_entries, parent=None):
            super(TableModel.SetEntriesCommand, self).__init__('edit XPL source for {}'.format(table.name), parent)
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

    def _set_entries(self, new_entries, undoable=True):
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
        self.endcomments = []
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
        super().mark_info_invalid()

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
            return '\n'.join([str(err) for err in self.info_by_row.get(index.row(), [])
                              if err.has_connection('cols', index.column())])
        if role == Qt.DecorationRole: #Qt.BackgroundColorRole:   #maybe TextColorRole?
            max_level = -1
            c = index.column()
            for err in self.info_by_row.get(index.row(), []):
                if err.has_connection('cols', c, c == 0):
                    # c == 0 -> whole row messages have decoration only in the first column
                    if err.level > max_level: max_level = err.level
            return info.info_level_icon(max_level)
            #c = QPalette().color(QPalette.Window)    #default color
            #if max_level == info.Info.ERROR: return QColor(255, 220, 220)
            #if max_level == info.Info.WARNING: return QColor(255, 255, 160)
            #if max_level == info.Info.INFO: return QColor(220, 220, 255)

    def flags(self, index):
        flags = super().flags(index) | Qt.ItemIsSelectable | Qt.ItemIsEnabled

        if not self.is_read_only(): flags |= Qt.ItemIsEditable
        #flags |= Qt.ItemIsDragEnabled
        #flags |= Qt.ItemIsDropEnabled

        return flags
