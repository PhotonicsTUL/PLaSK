from ..qt import QtCore

class ETreeModel(QtCore.QAbstractItemModel):
    """
        Implementation of QtCore.QAbstractItemModel which display etree Element.
        By default only tag is displayed, but subclasses can override columnCount, headerData and data to change this.
    """

    def __init__(self, root, parent=None):
        super(ETreeModel, self).__init__(parent)
        self.root = root

    def columnCount(self, parent):
        return 1
    #    if parent.isValid():
    #        return parent.internalPointer().columnCount()
    #    else:
    #        return self.root.columnCount()

    def data(self, index, role):
        if not index.isValid(): return None
        if role != QtCore.Qt.DisplayRole: return None
        item = index.internalPointer()
        return item.tag

    def flags(self, index):
        if not index.isValid(): return QtCore.Qt.NoItemFlags
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return 'tag'
        return None

    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent): return QtCore.QModelIndex()

        if not parent.isValid():
            parentItem = self.root
        else:
            parentItem = parent.internalPointer()
        childItem = parentItem.get(row, None)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid(): return QtCore.QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.getparent()
        if parentItem == self.root: return QtCore.QModelIndex()
        return self.createIndex(parentItem.index(childItem), 0, parentItem)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.root
        else:
            parentItem = parent.internalPointer()

        return len(parentItem)