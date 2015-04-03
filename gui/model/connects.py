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

from ..qt import QtCore
from lxml import etree

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
]


class ConnectsModel(TableModel):

    class Entry:
        def __init__(self, output, input, comment=None):
            self.output = output
            self.input = input
            self.comment = comment

    def __init__(self, parent=None, info_cb=None, *args):
        TableModel.__init__(self, 'connects', parent, info_cb, *args)

    def set_xml_element(self, element, undoable=True):
        new_entries = []
        with OrderedTagReader(element) as r:
            for e in r.iter("connect"):
                with AttributeReader(e) as a:
                    new_entries.append(ConnectsModel.Entry(a.get("out", ""), a.get("in", "")))
        self._set_entries(new_entries, undoable)

    # XML element that represents whole section
    def get_xml_element(self):
        res = etree.Element(self.name)
        for e in self.entries:
            if e.comment: res.append(etree.Comment(e.comment))
            etree.SubElement(res, "connect", { "out": e.output, "in": e.input })  #.tail = '\n'
        return res

    def get(self, col, row):
        if col == 0: return self.entries[row].output
        if col == 1: return self.entries[row].input
        if col == 2: return self.entries[row].comment
        raise IndexError('column number for ConnectsModel should be 0, 1, or 2, but is %d' % col)

    def set(self, col, row, value):
        if col == 0: self.entries[row].output = value
        elif col == 1: self.entries[row].input = value
        elif col == 2: self.entries[row].comment = value
        else: raise IndexError('column number for ConnectsModel should be 0, 1, or 2, but is %d' % col)

    def create_default_entry(self):
        return ConnectsModel.Entry("out", "in")

    # QAbstractListModel implementation

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2    # 3 if comment supported

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'Provider'
            if col == 1: return 'Receiver'
            if col == 2: return 'Comment'
        return None

    def create_info(self):
        res = super(ConnectsModel, self).create_info()
        for i, d in enumerate(self.entries):
            if not d.output:
                res.append(Info('Connection output is required [row: {}]'.format(i+1), Info.ERROR, rows=[i], cols=[0]))
            if not d.input:
                res.append(Info('Connection input is required [row: {}]'.format(i+1), Info.ERROR, rows=[i], cols=[1]))
        return res
