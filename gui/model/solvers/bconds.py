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

from ...qt.QtCore import *

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str


class SchemaBoundaryConditions(object):

    def __init__(self, name, label, group, mesh_type, mesh_attr=None, geometry_attr=None, values=None):
        self.geometry_attr = geometry_attr
        self.mesh_attr = mesh_attr
        self.name = name
        if group is None:
            self.label = "Boundary Conditions"
        else:
            self.label = group + " Boundary Conditions"
        self.label2 = label
        self.mesh_type = mesh_type
        self.keys = ('value',) if values is None else values

    def to_xml(self, conditions):
        pass

    def from_xml(self, element):
        return []

    def create_default_entry(self):
        pass

    def create_place(self, Label):
        pass


class RectangularBC(SchemaBoundaryConditions):

    class PlaceNode(object):
        def __init__(self, parent=None):
            self.parent = parent
            self.children = []

    class PlaceSide(PlaceNode):
        def __init__(self, parent, side, of=None, path=None):
            super(RectangularBC.PlaceSide, self).__init__(parent)
            self.side = side
            self.object = of
            self.path = path

        def get_xml_element(self):
            place = etree.Element('place')
            place.attrib['side'] = self.side
            if self.object is not None:
                place.attrib['object'] = self.object
            if self.path is not None:
                place.attrib['path'] = self.path
            return place

        def copy_from(self, old):
            self.parent = old.parent
            self.object = old.object
            self.path = old.path

        @property
        def label(self):
            return self.side.title()

        def __eq__(self, other):
            return type(other) == RectangularBC.PlaceSide and \
                   self.side == other.side and self.object == other.object and self.path == other.path

        def __str__(self):
            if self.object is None: return ''
            if self.path is None:
                return "<i>Object:</i>&nbsp;" + self.object
            else:
                return "<i>Object:</i>&nbsp;" + self.object + "&nbsp;&nbsp;&nbsp;&nbsp;<i>Path:</i>&nbsp;" + self.path

    class PlaceLine(PlaceNode):
        def __init__(self, parent, line, at=0, start=0, stop=0):
            super(RectangularBC.PlaceLine, self).__init__(parent)
            self.line = line
            self.at = at
            self.start = start
            self.stop = stop

        def get_xml_element(self):
            place = etree.Element('place')
            place.attrib['line'] = self.line
            place.attrib['at'] = str(self.at)
            place.attrib['start'] = str(self.start)
            place.attrib['stop'] = str(self.stop)
            return place

        def copy_from(self, old):
            self.parent = old.parent
            self.at = old.at
            self.start = old.start
            self.stop = old.stop

        @property
        def label(self):
            return self.line.title() + ' Line'

        def __eq__(self, other):
            return type(other) == RectangularBC.PlaceLine and \
                   self.line == other.line and self.at == other.at and \
                   self.start == other.start and self.stop == other.stop

        def __str__(self):
            return "<i>Pos:</i>&nbsp;{0.at}&nbsp;&nbsp;&nbsp;&nbsp;" \
                   "<i>From:</i>&nbsp;{0.start}&nbsp;&nbsp;&nbsp;&nbsp;<i>To:</i>&nbsp;{0.stop}".format(self)

    class SetOp(PlaceNode):
        def __init__(self, parent, operation=None, children=None):
            super(RectangularBC.SetOp, self).__init__(parent)
            self.operation = operation
            self.children = [] if children is None else children
            self._fix_parents_of_children()

        def _fix_parents_of_children(self):
            for c in self.children:
                if c is not None:
                    c.parent = self

        def get_xml_element(self):
            place = etree.Element(self.operation)
            for child in self.children:
                if child is not None:
                    place.append(child.get_xml_element())
            return place

        def copy_from(self, old):
            self.parent = old.parent
            self.operation = old.operation
            self.children = old.children
            self._fix_parents_of_children()

        @property
        def label(self):
            return self.operation.title()

        def __eq__(self, other):
            return type(other) == RectangularBC.SetOp and \
                   self.operation == other.operation and \
                   all(c == oc for c, oc in zip(self.children, other.children))

        def child_label(self, index):
            return self.children[index].label if index < len(self.children) and self.children[index] is not None else '?'

        def __str__(self):
            return "<i>of</i>&nbsp;{0}&nbsp;<i>and</i>&nbsp;{1}".format(self.child_label(0), self.child_label(1))


    @staticmethod
    def place_from_xml(place):
        if place.tag in ('intersection', 'union', 'difference'):
            return RectangularBC.SetOp(place.tag, (RectangularBC.place_from_xml(el) for el in list(place)[:2]))
        else:   # place tag:
            # TODO ensure that: place.tag == 'place'
            side = place.attrib.get('side')
            line = place.attrib.get('line')
            if side is not None and line is not None:
                raise TypeError("'side' and 'line' attributes cannot be specified simultaneously")
            if side is not None:
                of = place.attrib.get('object')
                path = place.attrib.get('path')
                return RectangularBC.PlaceSide(side, of, path)
            elif line is not None:
                at = place.attrib['at']
                start = place.attrib['start']
                stop = place.attrib['stop']
                return RectangularBC.PlaceLine(line, at, start, stop)
            else:
                raise TypeError("Exactly one of 'side' and 'line' attributes must be given")

    @staticmethod
    def place_from_xml_cond_tag(cond_element):
        try:
            side = cond_element.attrib['place']
        except KeyError:
            RectangularBC.place_from_xml(next(cond_element.iter()))
        else:
            return RectangularBC.PlaceSide(side)

    def to_xml(self, conditions):
        if conditions:
            element = etree.Element(self.name)
            for place, value in conditions:
                cond = etree.Element('condition')
                cond.append(place.get_xml_element())
                for key in self.keys:
                    val = value[key]
                    if val is not None:
                        cond.attrib[key] = val
                element.append(cond)
            return element

    def from_xml(self, element):
        conditions = []
        for cond in element.findall('condition'):
            place = self.place_from_xml_cond_tag(cond)
            value = dict((key, cond.attrib.get(key)) for key in self.keys)
            conditions.append((place, value))
        return conditions

    def create_default_entry(self):
        return RectangularBC.PlaceSide(None, 'left'), dict((key,None) for key in self.keys)

    def create_place(self, label, parent=None):
        return {
            "Left": lambda: RectangularBC.PlaceSide(parent, 'left'),
            "Right": lambda: RectangularBC.PlaceSide(parent, 'right'),
            "Top": lambda: RectangularBC.PlaceSide(parent, 'top'),
            "Bottom": lambda: RectangularBC.PlaceSide(parent, 'bottom'),
            "Horizontal Line": lambda: RectangularBC.PlaceLine(parent, 'horizontal'),
            "Vertical Line": lambda: RectangularBC.PlaceLine(parent, 'vertical'),
            "Front": lambda: RectangularBC.PlaceSide(parent, 'front'),
            "Back": lambda: RectangularBC.PlaceSide(parent, 'back'),
        }[label]()


class BoundaryConditionsModel(QAbstractItemModel):

    def __init__(self, schema, conditions=None, parent=None):
        super(BoundaryConditionsModel, self).__init__(parent)
        self.schema = schema
        if conditions is None:
            self.entries = []
        else:
            self.entries = conditions

    def children_of(self, index):
        return index.internalPointer().children if index is not None and index.isValid() else self.entries

    def node_for_index(self, index):
        return index.internalPointer() if index.isValid() else self

    def index_for_node(self, node, column=0):
        if node is None or isinstance(node, BoundaryConditionsModel):
            return QModelIndex()
        try:
            c = self.node.parent.children if node.parent else self.entries
            child_row = 0
            for row, item in enumerate(c):
                if item is node:
                    child_row = row
                    break
            index = self.createIndex(child_row, column, node)
        except (ValueError, IndexError, TypeError):
            return QModelIndex()
        return index

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0: return 0
        return len(self.children(parent))

    def columnCount(self, parent=QModelIndex()):
        return 2 + len(self.schema.keys)

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent): return QModelIndex()
        return self.createIndex(row, column, self.children(parent)[row]) #if 0 <= row < len(l) else QModelIndex()

    def parent(self, index):
        if not index.isValid(): return QModelIndex()
        return self.index_for_node(index.internalPointer().parent)

    def headerData(self, no, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if no == 0: return 'Place'
                elif no == 1: return 'Place Details'
                else: return self.schema.keys[no - 2].title()
            elif orientation == Qt.Vertical:
                return str(no)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            try:
                place, values = self.entries[index.row()]
            except IndexError:
                return
            col = index.column()
            if col == 0:
                return place.label
            elif col == 1:
                return str(place)
            else:
                return values[self.schema.keys[col-2]]
        # if role == Qt.ToolTipRole:
        #     return '\n'.join([str(err) for err in self.info_by_row.get(index.row(), []) if err.has_connection('cols', index.column())])
        # if role == Qt.DecorationRole: #Qt.BackgroundColorRole:   # maybe TextColorRole?
        #     max_level = -1
        #     c = index.column()
        #     for err in self.info_by_row.get(index.row(), []):
        #         if err.has_connection('cols', c, c == 0):   # c == 0 -> whole row messages have decoration only in first column
        #             if err.level > max_level: max_level = err.level
        #     return info.info_level_icon(max_level)

    def setData(self, index, value, role=Qt.EditRole):
        row = index.row()
        if row < 0 or row >= len(self.entries):
            return False
        col = index.column()
        if col == 0:
            old_place = self.entries[row][0]
            new_place = self.schema.create_place(value)
            self.entries[row] = new_place, self.entries[row][1]
            if type(old_place) == type(new_place):
                new_place.copy_from(old_place)
            self.dataChanged.emit(index, index)
        elif col == 1:
            return False
        else:
            self.entries[row][1][self.schema.keys[col-2]] = value
            self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        flags = super(BoundaryConditionsModel, self).flags(index) \
                | Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        return flags

    def insert(self, index=None, value=None):
        if value is None:
            value = self.schema.create_default_entry()
            if value is None: return
        if index is None or index < 0: index = 0
        if index >= len(self.entries): index = len(self.entries)
        self.beginInsertRows(QModelIndex(), index, index)
        self.entries.insert(index, value)
        self.endInsertRows()
        return index

    def remove(self, index=None):
        if index is None or index < 0 or index >= len(self.entries):
            return
        self.beginRemoveRows(QModelIndex(), index, index)
        del self.entries[index]
        self.endRemoveRows()
        return index

    def swap_entries(self, index1, index2):
        self.beginMoveRows(QModelIndex(), index2, index2, QModelIndex(), index1)
        self.entries[index1], self.entries[index2] = self.entries[index2], self.entries[index1]
        self.endMoveRows()


BCONDS = {
    'Rectangular2D': RectangularBC,
    'Rectangular3D': RectangularBC,
}