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

from qt import QtCore
from lxml import etree

from .. import SectionModel
from .reader import GNReadConf
from ...utils.xml import AttributeReader
from .types import construct_geometry_object, geometry_types_geometries

__author__ = 'qwak'

class GeometryModel(QtCore.QAbstractItemModel, SectionModel):

    def __init__(self, parent=None, info_cb=None, *args):
        SectionModel.__init__(self, 'geometry', info_cb)
        QtCore.QAbstractItemModel.__init__(self, parent, *args)
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
        with AttributeReader(element) as a:
            new_axes = a.get('axes')
        conf = GNReadConf(axes = new_axes)
        new_roots = []
        for child_element in element:
            new_roots.append(construct_geometry_object(child_element, conf, geometry_types_geometries))
        self.modelAboutToBeReset.emit()
        self.axes = new_axes
        self.roots = new_roots
        self.modelReset.emit()
        self.fire_changed()

    # QAbstractItemModel implementation
    def columnCount(self, parent):
        return 2

    def data(self, index, role):
        if not index.isValid(): return None
        if role != QtCore.Qt.DisplayRole: return None
        item = index.internalPointer()
        if index.column() == 0:
            return item.tag_name(full_name=True)
        else:
            return getattr(item, 'name', '')

    def flags(self, index):
        if not index.isValid(): return QtCore.Qt.NoItemFlags
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return ('tag', 'name')[section]
        return None

    def _children_list(self, parent_index):
        return parent_index.internalPointer().children if parent_index.isValid() else self.roots

    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent): return QtCore.QModelIndex()
        l = self._children_list(parent)
        return self.createIndex(row, column, l[row]) if row < len(l) else QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid(): return QtCore.QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.parent
        if parentItem is None: return QtCore.QModelIndex()
        return self.createIndex(parentItem.children.index(childItem), 0, parentItem)

    def rowCount(self, parent):
        return len(self._children_list(parent))
