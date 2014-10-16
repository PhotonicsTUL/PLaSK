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

from lxml import etree
from ...qt import QtCore

from .. import SectionModel
from .reader import GNReadConf
from .constructor import construct_geometry_object
from ...utils.str import none_to_empty
from ...utils.xml import AttributeReader
from .types import geometry_types_geometries

import sys

__author__ = 'qwak'

class GeometryModel(QtCore.QAbstractItemModel, SectionModel):

    def __init__(self, parent=None, info_cb=None, *args):
        QtCore.QAbstractItemModel.__init__(self, parent, *args)
        SectionModel.__init__(self, 'geometry', info_cb)
        #TableModelEditMethods.__init__(self)
        self.roots = []
        self.axes = None

    # XML element that represents whole section
    def get_xml_element(self):
        res = etree.Element(self.name)
        if self.axes: res.attrib['axes'] = self.axes
        conf = GNReadConf(axes = self.axes)
        for geom in self.roots: res.append(geom.get_xml_element(conf))
        return res

    def set_xml_element(self, element):
        with AttributeReader(element) as a: new_axes = a.get('axes')
        conf = GNReadConf(axes = new_axes)
        new_roots = []
        for child_element in element:
            new_roots.append(construct_geometry_object(child_element, conf, geometry_types_geometries))
        self.beginResetModel()
        self.axes = new_axes
        self.roots = new_roots
        self.endResetModel()
        self.fire_changed()

    # QAbstractItemModel implementation:
    def columnCount(self, parent = QtCore.QModelIndex()):
        sys.stderr.write('columnCount {} {}\n'.format(parent.isValid(), parent.internalPointer()))
        return 2

    def data(self, index, role = QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        if role != QtCore.Qt.DisplayRole: return None
        sys.stderr.write('data {} {}\n'.format(index.isValid(), index.internalPointer()))
        item = index.internalPointer()
        if item is None:
            sys.stderr.write('ITEM IS NONE\n')
        if index.column() == 0:
            return item.tag_name(full_name=True)
        else:
            return none_to_empty(getattr(item, 'name', ''))

    def flags(self, index):
        sys.stderr.write('flags {} {}\n'.format(index.isValid(), index.internalPointer()))
        if not index.isValid(): return QtCore.Qt.NoItemFlags
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def headerData(self, section, orientation, role = QtCore.Qt.DisplayRole):
        sys.stderr.write('headerData {}\n'.format(section))
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return ('tag', 'name')[section]
        return None

    def _children_list(self, parent_index):
        #sys.stderr.write('parent is valid: {} {}\n'.format(parent_index.isValid(), parent_index.internalPointer()))
        return parent_index.internalPointer().children if parent_index.isValid() else self.roots

    def index(self, row, column, parent = QtCore.QModelIndex()):
        sys.stderr.write('index {} {} {} {}\n'.format(parent.isValid(), parent.internalPointer(), row, column))
        if not self.hasIndex(row, column, parent):
            sys.stderr.write(' self.hasIndex returns False\n')
            return QtCore.QModelIndex()
        l = self._children_list(parent)
        res = self.createIndex(row, column, l[row]) #if 0 <= row < len(l) else QtCore.QModelIndex()
        sys.stderr.write(' is {} {} {} {}\n'.format(res.isValid(), res.internalPointer(), res.row(), res.column()))
        return self.createIndex(row, column, l[row]) #if 0 <= row < len(l) else QtCore.QModelIndex()
        #if row < len(l):
        #    return self.createIndex(row, column, l[row])
        #else:
        #    QtCore.QModelIndex()


    def parent(self, index):
        sys.stderr.write('parent {} {}\n'.format(index.isValid(), index.internalPointer()))
        if not index.isValid(): return QtCore.QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.parent
        if parentItem is None: return QtCore.QModelIndex()
        return self.createIndex(parentItem.children.index(childItem), 0, parentItem)

    def rowCount(self, parent = QtCore.QModelIndex()):
        sys.stderr.write('rowCount {} {}\n'.format(parent.isValid(), parent.internalPointer()))
        if parent.column() > 0: return 0
        return len(self._children_list(parent))
