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

from ...qt.QtCore import Qt, QAbstractTableModel, QModelIndex

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

    def __init__(self, name, label, mesh_type, values=None):
        self.name = name
        self.label = label + ' Boundary Conditions'
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

    class PlaceSide(object):
        def __init__(self, side, of=None, path=None):
            self.side = side
            self.object = of
            self.path = path
        def get_xml_element(self):
            element = etree.Element('condition')
            if self.object is None:
                element.attrib['place'] = self.side
            else:
                place = etree.SubElement(element, 'place')
                place.attrib['side'] = self.side
                place.attrib['object'] = self.object
                if self.path is not None:
                    place.attrib['path'] = self.path
            return element
        def copy_from(self, old):
            self.object = old.object
            self.path = old.path
        @property
        def label(self):
            return self.side.title()
        def __eq__(self, other):
            return self.side == other.side and self.object == other.object and self.path == other.path
        def __str__(self):
            if self.object is None: return ''
            if self.path is None:
                return "<i>Object:</i>&nbsp;" + self.object
            else:
                return "<i>Object:</i>&nbsp;" + self.object + "&nbsp;&nbsp;&nbsp;&nbsp;<i>Path:</i>&nbsp;" + self.path

    class PlaceLine(object):
        def __init__(self, line, at=0, start=0, stop=0):
            self.line = line
            self.at = at
            self.start = start
            self.stop = stop
        def get_xml_element(self):
            element = etree.Element('condition')
            place = etree.SubElement(element, 'place')
            place.attrib['line'] = self.line
            place.attrib['at'] = self.at
            place.attrib['start'] = self.start
            place.attrib['stop'] = self.stop
            return element
        def copy_from(self, old):
            self.at = old.at
            self.start = old.start
            self.stop = old.stop
        @property
        def label(self):
            return self.line.title() + ' Line'
        def __eq__(self, other):
            return self.line == other.line and self.at == other.at and \
                   self.start == other.start and self.stop== other.stop
        def __str__(self):
            return "<i>Pos:</i>&nbsp;{0.at}&nbsp;&nbsp;&nbsp;&nbsp;" \
                   "<i>From:</i>&nbsp;{0.start}&nbsp;&nbsp;&nbsp;&nbsp;<i>To:</i>&nbsp;{0.stop}".format(self)

    @staticmethod
    def place_from_xml(element):
        try:
            side = element.attrib['place']
        except KeyError:
            place = element.find('place')
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
        else:
            return RectangularBC.PlaceSide(side)

    def to_xml(self, conditions):
        element = etree.Element(self.name)
        for place,value in conditions:
            cond = place.get_xml_element()
            for key in self.keys:
                val = value[key]
                if val is not None:
                    cond.attrib[key] = val
            element.append(cond)
        return element

    def from_xml(self, element):
        conditions = []
        for cond in element.findall('condition'):
            place = self.place_from_xml(cond)
            value = dict((key, cond.attrib.get(key)) for key in self.keys)
            conditions.append((place, value))
        return conditions

    def create_default_entry(self):
        return RectangularBC.PlaceSide('left'), dict((key,None) for key in self.keys)

    def create_place(self, label):
        return {
            "Left": lambda: RectangularBC.PlaceSide('left'),
            "Right": lambda: RectangularBC.PlaceSide('right'),
            "Top": lambda: RectangularBC.PlaceSide('top'),
            "Bottom": lambda: RectangularBC.PlaceSide('bottom'),
            "Horizontal Line": lambda: RectangularBC.PlaceLine('horizontal'),
            "Vertical Line": lambda: RectangularBC.PlaceLine('vertical'),
            "Front": lambda: RectangularBC.PlaceSide('front'),
            "Back": lambda: RectangularBC.PlaceSide('back'),
        }[label]()


class BoundaryConditionsModel(QAbstractTableModel):

    def __init__(self, schema, conditions=None, parent=None):
        super(BoundaryConditionsModel, self).__init__(parent)
        self.schema = schema
        if conditions is None:
            self.entries = []
        else:
            self.entries = conditions

    def rowCount(self, parent=QModelIndex()):
        return len(self.entries)

    def columnCount(self, parent=QModelIndex()):
        return 2 + len(self.schema.keys)

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if col == 0: return 'Place'
            elif col == 1: return 'Place Details'
            else: return self.schema.keys[col-2].title()

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
        # if role == Qt.DecorationRole: #Qt.BackgroundColorRole:   #maybe TextColorRole?
        #     max_level = -1
        #     c = index.column()
        #     for err in self.info_by_row.get(index.row(), []):
        #         if err.has_connection('cols', c, c == 0):   # c == 0 -> whole row messages hav decoration only in first column
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
        elif col == 1:
            return False
        else:
            self.entries[row][1][self.schema.keys[col-2]] = value
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