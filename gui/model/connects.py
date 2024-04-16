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

from ..qt.QtCore import *
from .table import TableModel
from .info import Info
from ..utils.xml import OrderedTagReader, AttributeReader

PROPS = [
    'Temperature',
    'HeatFlux',
    'Heat',
    'Potential',
    'CurrentDensity',
    'CarriersConcentration',
    'ElectronsConcentration',
    'HolesConcentration',
    'Gain',
    'GainOverCarriersConcentration',
    'LightMagnitude',
    'LightE',
    'LightH',
    'ThermalConductivity',
    'Conductivity',
    'RefractiveIndex',
    'Epsilon',
]


class ConnectsModel(TableModel):

    class Entry:
        def __init__(self, output, input, comments=None):
            self.output = output
            self.input = input
            self.comments = [] if comments is None else comments

    def __init__(self, parent=None, info_cb=None, *args):
        TableModel.__init__(self, 'connects', parent, info_cb, *args)

    def load_xml_element(self, element, undoable=True):
        new_entries = []
        with OrderedTagReader(element) as r:
            for e in r.iter("connect"):
                with AttributeReader(e) as a:
                    new_entries.append(ConnectsModel.Entry(a.get("out", ""), a.get("in", ""), e.comments))
            self.endcomments = r.get_comments()
        self._set_entries(new_entries, undoable)

    # XML element that represents whole section
    def make_xml_element(self):
        res = etree.Element(self.name)
        for e in self.entries:
            for c in e.comments:
                res.append(etree.Comment(c))
            etree.SubElement(res, "connect", { "out": e.output, "in": e.input })  #.tail = '\n'
        for c in self.endcomments:
            res.append(etree.Comment(c))
        return res

    def get(self, col, row):
        if col == 0: return self.entries[row].input
        if col == 1: return self.entries[row].output
        if col == 2: return self.entries[row].comments
        raise IndexError('column number for ConnectsModel should be 0, 1, or 2, but is %d' % col)

    def set(self, col, row, value):
        if   col == 0: self.entries[row].input = value
        elif col == 1: self.entries[row].output = value
        elif col == 2: self.entries[row].comments = value
        else: raise IndexError('column number for ConnectsModel should be 0, 1, or 2, but is %d' % col)

    def create_default_entry(self):
        return ConnectsModel.Entry("out", "in")

    # QAbstractListModel implementation

    def columnCount(self, parent=QModelIndex()):
        return 2    # 3 if comment supported

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if col == 0: return 'Receiver'
            if col == 1: return 'Provider'
            if col == 2: return 'Comment'
        return None

    def create_info(self):
        res = super().create_info()
        for i, d in enumerate(self.entries):
            if not d.input:
                res.append(Info('Connection input is required [row: {}]'.format(i+1), Info.ERROR, rows=[i], cols=[0]))
            if not d.output:
                res.append(Info('Connection output is required [row: {}]'.format(i+1), Info.ERROR, rows=[i], cols=[1]))
        return res
