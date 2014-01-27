from model.base import SectionModel
from PyQt4 import QtCore

class TableModel(QtCore.QAbstractTableModel, SectionModel):
   
    def __init__(self, name, parent=None, info_cb = None, *args):
        SectionModel.__init__(self, name, info_cb)
        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self.entries = []     
        
    # QAbstractListModel implementation
    def rowCount(self, parent = QtCore.QModelIndex()):
        if parent.isValid(): return 0
        return len(self.entries)
    
    def data(self, index, role = QtCore.Qt.DisplayRole): 
        if index.isValid() and (role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole): 
            return self.get(index.column(), index.row())
        return None
        
    def flags(self, index):
        flags = super(TableModel, self).flags(index)

        if not self.isReadOnly(): flags |= QtCore.Qt.ItemIsEditable
        flags |= QtCore.Qt.ItemIsSelectable
        flags |= QtCore.Qt.ItemIsEnabled
        #flags |= QtCore.Qt.ItemIsDragEnabled
        #flags |= QtCore.Qt.ItemIsDropEnabled

        return flags
    
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        self.set(index.column(), index.row(), value)
        self.dataChanged.emit(index, index)
        self.fireChanged()
        return True
    
    def insert(self, index = None, value = None):
        if self.isReadOnly(): return
        if not value: value = self.createDefaultEntry()
        if 0 <= index and index <= len(self.entries):
            self.beginInsertRows(QtCore.QModelIndex(), index, index)
            self.entries.insert(index, value)
        else:
            index = len(self.entries)
            self.beginInsertRows(QtCore.QModelIndex(), index, index)
            self.entries.append(value)
        self.endInsertRows()
        self.fireChanged()
        return index
    
    def remove(self, index):
        if self.isReadOnly() or index < 0 or index >= len(self.entries): return
        self.beginRemoveRows(QtCore.QModelIndex(), index, index)
        del self.entries[index]
        self.endRemoveRows()

    def swapNeighbourEntries(self, index1, index2):
        if self.isReadOnly(): return
        if index2 < index1: index1, index2 = index2, index1 
        self.beginMoveRows(QtCore.QModelIndex(), index2, index2, QtCore.QModelIndex(), index1)
        self.entries[index1], self.entries[index2] = self.entries[index2], self.entries[index1]
        self.endMoveRows()