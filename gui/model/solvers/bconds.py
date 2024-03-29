# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


from lxml import etree

from ...utils.xml import OrderedTagReader
from ...qt.QtCore import *
from ...qt.QtGui import *

basestring = str, bytes
class SchemaBoundaryConditions:

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
        self.endcomments = []

    def to_xml(self, conditions):
        pass

    def from_xml(self, element):
        return []

    def create_default_entry(self):
        pass

    def create_place(self, label):
        pass


class RectangularBC(SchemaBoundaryConditions):

    class PlaceNode:
        def __init__(self, place=None, value=None, comments=None):
            self.parent = None
            self.children = []
            self.place = place
            if value: self.value = value
            self.comments = []
            self.comments1 = comments if comments is not None else []
            self.comments2 = []

        def append_child(self, node):
            node.parent = self
            self.children.append(node)

        @property
        def children_places(self):
            for c in self.children:
                yield c.place

        @property
        def place(self):
            return self._place

        @place.setter
        def place(self, place):
            self._place = place
            if self._place is not None:
                self._place.node = self

        def child_place(self, index):
            return self.children[index].place

        def to_xml(self):
            result = self.place.make_xml_element()
            for child in self.children:
                for c in child.comments1:
                    result.append(etree.Comment(c))
                result.append(child.to_xml())
                for c in child.comments2:
                    result.append(etree.Comment(c))
            return result


    class PlaceSide:

        required_child_count = 0
        is_editable = True

        def __init__(self, side, of=None, path=None):
            super().__init__()
            self.side = side
            self.object = of
            self.path = path

        def make_xml_element(self):
            place = etree.Element('place')
            place.attrib['side'] = self.side
            if self.object is not None:
                place.attrib['object'] = self.object
            if self.path is not None:
                place.attrib['path'] = self.path
            return place

        def copy_from(self, old):
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

    class PlaceLine:

        required_child_count = 0
        is_editable = True

        def __init__(self, line, at=0, start=0, stop=0):
            super().__init__()
            self.line = line
            self.at = at
            self.start = start
            self.stop = stop

        def make_xml_element(self):
            place = etree.Element('place')
            place.attrib['line'] = self.line
            place.attrib['at'] = str(self.at)
            place.attrib['start'] = str(self.start)
            place.attrib['stop'] = str(self.stop)
            return place

        def copy_from(self, old):
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

    class SetOp:

        required_child_count = 2
        is_editable = False

        def __init__(self, operation=None):
            super().__init__()
            self.operation = operation

        def make_xml_element(self):
            return etree.Element(self.operation)

        def copy_from(self, old):
            pass

        @property
        def label(self):
            return self.operation.title()

        def __eq__(self, other):
            return type(other) == RectangularBC.SetOp and \
                   self.operation == other.operation and \
                   all(c == oc for c, oc in zip(self.place.children_places, other.place.children_places))

        def child_label(self, index):
            return self.node.child_place(index).label if index < len(self.node.children) and self.node.child_place(index) is not None else '?'

        def __str__(self):
            return "<i>of</i>&nbsp;{0}&nbsp;<i>and</i>&nbsp;{1}".format(self.child_label(0), self.child_label(1))


    def place_node_from_xml(self, place_element, comments=None):
        if place_element.tag in ('intersection', 'union', 'difference'):
            place = RectangularBC.PlaceNode(RectangularBC.SetOp(place_element.tag), comments=comments)
            comments = []
            for el in place_element:
                if el.tag is etree.Comment:
                    comments.append(el.text)
                else:
                    place.append_child(self.place_node_from_xml(el, comments))
                    comments = []
            if len(place.children) > 2:
                raise ValueError("Exactly two places must be given inside {}".format(place_element.tag))
            for _ in range(2 - len(place.children)):  # we must have exactly two children
                place.append_child(self.create_default_entry())
                place.children[-1].comments1 = comments
                comments = []
            place.children[-1].comments2 = comments
            return place
        else:   # place tag:
            # TODO ensure that: place.tag == 'place'
            side = place_element.attrib.get('side')
            line = place_element.attrib.get('line')
            if side is not None and line is not None:
                raise TypeError("'side' and 'line' attributes cannot be specified simultaneously")
            if side is not None:
                of = place_element.attrib.get('object')
                path = place_element.attrib.get('path')
                return RectangularBC.PlaceNode(RectangularBC.PlaceSide(side, of, path), comments=comments)
            elif line is not None:
                at = place_element.attrib['at']
                start = place_element.attrib['start']
                stop = place_element.attrib['stop']
                return RectangularBC.PlaceNode(RectangularBC.PlaceLine(line, at, start, stop), comments=comments)
            else:
                raise ValueError("Exactly one of 'side' and 'line' attributes must be given")

    def place_node_from_xml_cond(self, cond_element):
        try:
            side = cond_element.attrib['place']
        except KeyError:
            contents = iter(cond_element)
            item = next(contents)
            comments = []
            while item.tag is etree.Comment:
                comments.append(item.text)
                item = next(contents)
            result = self.place_node_from_xml(item, comments)
            for item in contents:
                if item.tag is not etree.Comment:
                    raise ValueError("Exactly one place must be given")
                result.comments2.append(item.text)
            return result
        else:
            return RectangularBC.PlaceNode(RectangularBC.PlaceSide(side))

    def to_xml(self, conditions):
        if conditions:
            element = etree.Element(self.name)
            for node in conditions:
                for c in node.comments:
                    element.append(etree.Comment(c))
                cond = etree.SubElement(element, 'condition')
                for c in node.comments1:
                    cond.append(etree.Comment(c))
                cond.append(node.to_xml())
                for key in self.keys:
                    val = node.value[key]
                    if val is not None:
                        cond.attrib[key] = val
                for c in node.comments2:
                    cond.append(etree.Comment(c))
            for c in self.endcomments:
                element.append(etree.Comment(c))
            return element

    def from_xml(self, element):
        conditions = []
        with OrderedTagReader(element) as reader:
            for cond in reader.iter('condition'):
                place_node = self.place_node_from_xml_cond(cond)
                place_node.value = dict((key, cond.attrib.get(key)) for key in self.keys)
                place_node.comments = cond.comments
                conditions.append(place_node)
            self.endcomments = reader.get_comments()
        return conditions

    def create_default_entry(self):
        return RectangularBC.PlaceNode(
            place=RectangularBC.PlaceSide('top'),
            value=dict((key,None) for key in self.keys))

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
            "Union": lambda: RectangularBC.SetOp('union'),
            "Intersection": lambda: RectangularBC.SetOp('intersection'),
            "Difference": lambda: RectangularBC.SetOp('difference')
        }[label]()


class BoundaryConditionsModel(QAbstractItemModel):

    def __init__(self, schema, conditions=None, parent=None):
        super().__init__(parent)
        self.schema = schema
        self.entries = [] if conditions is None else conditions

    def children_of_index(self, index):
        return index.internalPointer().children if index is not None and index.isValid() else self.entries

    def children_of_node(self, node):
        return self.entries if node is None else node.children

    def place_for_index(self, index):
        return index.internalPointer() if index.isValid() else self

    def index_for_place(self, place_node, column=0):
        if place_node is None:  # root has no parent
            return QModelIndex()
        try:
            return self.createIndex(self.children_of_node(place_node.parent).index(place_node), column, place_node)
        except (ValueError, IndexError, TypeError):
            return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0: return 0
        return len(self.children_of_index(parent))

    def columnCount(self, parent=QModelIndex()):
        return 3 + len(self.schema.keys)

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent): return QModelIndex()
        return self.createIndex(row, column, self.children_of_index(parent)[row]) #if 0 <= row < len(l) else QModelIndex()

    def parent(self, index):
        if not index.isValid(): return QModelIndex()
        return self.index_for_place(index.internalPointer().parent)

    def headerData(self, no, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            # if orientation == Qt.Orientation.Horizontal:
                if no == 0: return 'Place'
                elif no == 1: return 'Place Details'
                elif no == 2: return ' #'
                else: return self.schema.keys[no-3].title()
            # elif orientation == Qt.Orientation.Vertical:
            #     return str(no)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            node = index.internalPointer()
            col = index.column()
            if col == 0:
                return node.place.label
            elif col == 1:
                return str(node.place)
            elif col == 2:
                if not index.parent().isValid():
                    return str(index.row()) + ' '
            else:
                try:
                    return node.value[self.schema.keys[col-3]]
                except (IndexError, AttributeError):
                    pass
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if index.column() == 2:
                return Qt.AlignmentFlag.AlignRight
        # elif role == Qt.ItemDataRole.BackgroundRole:
        #     if index.column() == 2:
        #         pal = QPalette()
        #         return pal.color(QPalette.ColorRole.Window)
        # elif role == Qt.ItemDataRole.ForegroundRole:
        #     if index.column() == 2:
        #         pal = QPalette()
        #         return pal.color(QPalette.ColorRole.WindowText)
        # if role == Qt.ItemDataRole.ToolTipRole:
        #     return '\n'.join([str(err) for err in self.info_by_row.get(index.row(), []) if err.has_connection('cols', index.column())])
        # if role == Qt.ItemDataRole.DecorationRole: #Qt.ItemDataRole.BackgroundColorRole:   # maybe TextColorRole?
        #     max_level = -1
        #     c = index.column()
        #     for err in self.info_by_row.get(index.row(), []):
        #         if err.has_connection('cols', c, c == 0):   # c == 0 -> whole row messages have decoration only in first column
        #             if err.level > max_level: max_level = err.level
        #     return info.info_level_icon(max_level)

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid(): return False
        node = index.internalPointer()
        entries = self.children_of_node(node.parent)
        row = index.row()
        if row < 0 or row >= len(entries):
            return False
        col = index.column()
        if col == 0:
            entry = entries[row]
            old_place = entry.place
            new_place = self.schema.create_place(value)
            entry.place = new_place
            if type(old_place) == type(new_place):
                new_place.copy_from(old_place)

            required_child_count = getattr(new_place, 'required_child_count', None)
            if required_child_count is not None:
                old_child_count = len(entry.children)
                if required_child_count > old_child_count:     # we need more children, add:
                    self.beginInsertRows(index, old_child_count, required_child_count-1)
                    for _ in range(old_child_count, required_child_count):
                        entry.append_child(self.schema.create_default_entry())
                    self.endInsertRows()
                elif required_child_count < old_child_count:   # some children are not needed, remove:
                    self.beginRemoveRows(index, required_child_count, old_child_count-1)
                    entry.children = entries[row].children[:required_child_count]
                    self.endRemoveRows()

            self.dataChanged.emit(index, index)
            return True
        if col == 1 or node.parent is not None:
            return False
        # col > 1 and node.parent is None:
        entries[row].value[self.schema.keys[col-3]] = value
        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        flags = super().flags(index) \
                | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled #| Qt.ItemFlag.ItemIsEditable
        col = index.column()
        if col == 0 or \
            (col == 1 and index.internalPointer().place.is_editable) or \
            (col > 2 and index.internalPointer().parent is None):
            flags |= Qt.ItemFlag.ItemIsEditable
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
        comments = self.entries[index].comments
        del self.entries[index]
        self.endRemoveRows()
        if len(comments) > 0:
            if index < len(self.entries):
                self.entries[index].comments = comments + self.entries[index].comments
            else:
                self.schema.endcomments = comments + self.schema.endcomments
        return index

    def swap_entries(self, index1, index2):
        self.beginMoveRows(QModelIndex(), index2, index2, QModelIndex(), index1)
        self.entries[index1], self.entries[index2] = self.entries[index2], self.entries[index1]
        self.endMoveRows()


BCONDS = {
    'Rectangular2D': RectangularBC,
    'Rectangular3D': RectangularBC,
}
