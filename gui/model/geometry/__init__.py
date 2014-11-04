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
import cgi

from lxml import etree
import operator
from ...qt import QtCore

from .. import SectionModel
from .reader import GNReadConf
from .constructor import construct_geometry_object, construct_by_name
from ...utils.str import none_to_empty
from ...utils.xml import AttributeReader
from .types import geometry_types_geometries

import sys

__author__ = 'qwak'


class GeometryModel(QtCore.QAbstractItemModel, SectionModel):

    def __init__(self, parent=None, info_cb=None):
        QtCore.QAbstractItemModel.__init__(self, parent)
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

    def stubs(self):
        res = 'class GEO(object):\n    """PLaSK object containing the defined geometry objects."""\n'
        res += '\n'.join(s for s in (r.stub() for r in self.roots) if s)
        return res

    # QAbstractItemModel implementation:
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 2

    def data(self, index, role = QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole: #or role == QtCore.Qt.EditRole:
            item = index.internalPointer()
            if index.column() == 0:
                return item.tag_name(full_name=True)
            else:
                name = getattr(item, 'name', '')
                if name:
                    res = '<span style="color: #866">name</span> <b>{}</b>'.format(cgi.escape(name))
                else:
                    res = ''
                for prop_table in (item.major_properties(), item.minor_properties()):
                    sorted_prop = sorted(prop_table, key=operator.itemgetter(0))
                    for n, v in sorted_prop:
                        if v is None: continue
                        if res: res += ' &nbsp; '
                        res += '<span style="color: #766">{}</span>&nbsp;{}'.format(cgi.escape(n).replace(' ', '&nbsp;'), cgi.escape(v).replace(' ', '&nbsp;'))
                        #replacing ' ' to '&nbsp;' is for better line breaking (not in middle of name/value)
                return res

    def flags(self, index):
        if not index.isValid(): return QtCore.Qt.NoItemFlags
        res = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        #if not self.is_read_only():
        #    if index.column() == 1 and hasattr(index.internalPointer(), 'name'): #name
        #        res |= QtCore.Qt.ItemIsEditable
        return res

    def headerData(self, section, orientation, role = QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return ('tag', 'properties')[section]
        return None

    def children_list(self, parent_index):
        return parent_index.internalPointer().children if parent_index.isValid() else self.roots

    def index(self, row, column, parent = QtCore.QModelIndex()):
        if not self.hasIndex(row, column, parent): return QtCore.QModelIndex()
        l = self.children_list(parent)
        return self.createIndex(row, column, l[row]) #if 0 <= row < len(l) else QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid(): return QtCore.QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.parent
        if parentItem is None: return QtCore.QModelIndex()
        return self.createIndex(parentItem.children.index(childItem), 0, parentItem)

    def rowCount(self, parent = QtCore.QModelIndex()):
        if parent.column() > 0: return 0
        return len(self.children_list(parent))

    def set(self, col, element, value):
        if col == 1:
            element.name = value
            return True
        return False

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid(): return False
        if self.set(index.column(), index.internalPointer(), value):
            self.fire_changed()
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    def removeRows(self, row, count, parent = QtCore.QModelIndex()):
        l = self.children_list(parent)
        end = row + count
        if row < 0 or end > len(l): return False
        self.beginRemoveRows(parent, row, end)
        del l[row:end]
        self.endRemoveRows()
        self.fire_changed()
        return True

    # other actions:
    def append_geometry(self, type_name):
        self.beginInsertRows(QtCore.QModelIndex(), len(self.roots), len(self.roots))
        self.roots.append(construct_by_name(type_name, geometry_types_geometries))
        self.endInsertRows()
        self.fire_changed()

    def _swap_neighbour_nodes(self, parent_index, row1, row2):
        if self.is_read_only(): return
        if row2 < row1: row1, row2 = row2, row1
        children = self.children_list(parent_index)
        if row1 < 0 or row2 >= len(children): return
        self.beginMoveRows(parent_index, row2, row2, parent_index, row1)
        children[row1], children[row2] = children[row2], children[row1]
        self.endMoveRows()
        self.fire_changed()

    def move_node_up(self, index):
        if not index.isValid(): return
        r = index.row()
        self._swap_neighbour_nodes(index.parent(), r-1, r)

    def move_node_down(self, index):
        if not index.isValid(): return
        r = index.row()
        self._swap_neighbour_nodes(index.parent(), r, r+1)

    def can_move_node_up_down(self, index):
        if not index.isValid(): return False, False
        children = self.children_list(index.parent())
        r = index.row()
        return r > 0, r+1 < len(children)