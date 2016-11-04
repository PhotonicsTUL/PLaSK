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


class ETreeModel(QAbstractItemModel):
    """
        Implementation of QAbstractItemModel which display etree Element.
        By default only tag is displayed, but subclasses can override columnCount, headerData and data to change this.
    """

    def __init__(self, root, parent=None):
        """
        :param root: ElementTree or sth. that has element attribute which is ElementTree
        :param parent: Qt parent
        """
        super(ETreeModel, self).__init__(parent)
        self._root = root

    @property
    def root(self):
        return getattr(self._root, 'element', self._root)

    def columnCount(self, parent):
        return 1
    #    if parent.isValid():
    #        return parent.internalPointer().columnCount()
    #    else:
    #        return self.root.columnCount()

    def data(self, index, role):
        if not index.isValid(): return None
        if role != Qt.DisplayRole: return None
        item = index.internalPointer()
        return item.tag

    def flags(self, index):
        if not index.isValid(): return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return 'tag'
        return None

    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent): return QModelIndex()

        if not parent.isValid():
            parentItem = self.root
        else:
            parentItem = parent.internalPointer()
        if row < len(parentItem):
            return self.createIndex(row, column, parentItem[row])
        else:
            return QModelIndex()


    def parent(self, index):
        if not index.isValid(): return QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.getparent()
        if parentItem == self.root: return QModelIndex()
        return self.createIndex(parentItem.index(childItem), 0, parentItem)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.root
        else:
            parentItem = parent.internalPointer()

        return len(parentItem)