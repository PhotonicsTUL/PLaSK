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
from collections import OrderedDict

from ...qt.QtCore import Qt

from ...qt import QtGui, QtCore
from ...utils.str import none_to_empty, empty_to_none
from ...utils.widgets import HTMLDelegate, ComboBox, table_edit_shortcut
from ..defines import DefinesCompletionDelegate, get_defines_completer
from ..table import table_with_manipulators
from ...model.solvers.bconds import RectangularBC, BoundaryConditionsModel


class PlaceDetailsEditor(QtGui.QWidget):
    pass


class RectangularPlaceSide(PlaceDetailsEditor):
    Model = RectangularBC.PlaceSide

    def __init__(self, controller, defines=None, parent=None):
        super(RectangularPlaceSide, self).__init__(parent)
        self.controller = controller
        self.setAutoFillBackground(True)
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 4, 0)
        self.object = ComboBox()
        self.object.sizePolicy().setHorizontalStretch(2)
        self.object.sizePolicy().setHorizontalPolicy(QtGui.QSizePolicy.MinimumExpanding)
        self.object.setEditable(True)
        self.path = ComboBox()
        self.path.sizePolicy().setHorizontalStretch(1)
        self.path.sizePolicy().setHorizontalPolicy(QtGui.QSizePolicy.MinimumExpanding)
        self.path.setEditable(True)
        if defines is not None:
            self.object.setCompleter(defines)
            self.path.setCompleter(defines)
        label = QtGui.QLabel(" Obj&ect:")
        label.setBuddy(self.object)
        label.setFixedWidth(label.fontMetrics().width(label.text()))
        layout.addWidget(label)
        layout.addWidget(self.object)
        label = QtGui.QLabel(" &Path:")
        label.setBuddy(self.path)
        label.setFixedWidth(label.fontMetrics().width(label.text()))
        layout.addWidget(label)
        layout.addWidget(self.path)
        self.setLayout(layout)

    def showEvent(self, event):
        super(RectangularPlaceSide, self).showEvent(event)
        self.object.setFocus()

    def load_data(self, data):
        self.object.clear()
        self.object.addItems([''] + list(self.controller.document.geometry.model.names()))
        self.object.setEditText(none_to_empty(data.object))
        self.path.clear()
        self.path.addItems([''] + list(self.controller.document.geometry.model.paths()))
        self.path.setEditText(none_to_empty(data.path))

    def save_data(self, data):
        data.object = empty_to_none(self.object.currentText())
        data.path = empty_to_none(self.path.currentText())


class RectangularPlaceLine(PlaceDetailsEditor):
    Model = RectangularBC.PlaceLine

    def __init__(self, controller, defines=None, parent=None):
        super(RectangularPlaceLine, self).__init__(parent)
        self.controller = controller
        self.setAutoFillBackground(True)
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(4, 0, 4, 0)
        self.position = QtGui.QLineEdit()
        self.start = QtGui.QLineEdit()
        self.stop = QtGui.QLineEdit()
        self.position.sizePolicy().setHorizontalStretch(1)
        self.start.sizePolicy().setHorizontalStretch(1)
        self.stop.sizePolicy().setHorizontalStretch(1)
        if defines is not None:
            self.position.setCompleter(defines)
            self.start.setCompleter(defines)
            self.stop.setCompleter(defines)
        label = QtGui.QLabel(" &Pos:")
        label.setBuddy(self.position)
        label.setFixedWidth(label.fontMetrics().width(label.text()))
        layout.addWidget(label)
        layout.addWidget(self.position)
        label = QtGui.QLabel(" &From:")
        label.setBuddy(self.start)
        label.setFixedWidth(label.fontMetrics().width(label.text()))
        layout.addWidget(label)
        layout.addWidget(self.start)
        label = QtGui.QLabel(" &To:")
        label.setBuddy(self.stop)
        label.setFixedWidth(label.fontMetrics().width(label.text()))
        layout.addWidget(label)
        layout.addWidget(self.stop)
        self.setLayout(layout)

    def showEvent(self, event):
        super(RectangularPlaceLine, self).showEvent(event)
        self.position.setFocus()

    def load_data(self, data):
        self.position.setText(str(none_to_empty(data.at)))
        self.start.setText(str(none_to_empty(data.start)))
        self.stop.setText(str(none_to_empty(data.stop)))

    def save_data(self, data):
        data.at = empty_to_none(self.position.text())
        data.start = empty_to_none(self.start.text())
        data.stop = empty_to_none(self.stop.text())


class BoundaryConditionsDialog(QtGui.QDialog):

    def __init__(self, controller, schema, data, parent=None):
        super(BoundaryConditionsDialog, self).__init__(parent)
        self.setWindowTitle(schema.label)

        self.schema = schema
        self.data = data

        self.table = QtGui.QTableView()
        model = BoundaryConditionsModel(schema, data)
        self.table.setModel(model)
        cols = model.columnCount(None)  # column widths:
        self.table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 250)
        self.table.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)

        table_edit_shortcut(self.table, 0, QtGui.QKeySequence(Qt.Key_P))
        table_edit_shortcut(self.table, 1, QtGui.QKeySequence(Qt.Key_D))
        used_shortcuts = ['p', 'd']

        self.defines_delegate = DefinesCompletionDelegate(controller.document.defines.model, self.table)
        defines_completer = get_defines_completer(controller.document.defines.model, self)
        for i in range(2, model.columnCount()):
            self.table.setColumnWidth(i, 150)
            self.table.setItemDelegateForColumn(i, self.defines_delegate)
            label = schema.keys[i-2].lower()
            for l in label:
                if l not in used_shortcuts:
                    table_edit_shortcut(self.table, i, QtGui.QKeySequence(l))
                    used_shortcuts.append(l)
                    break

        self.place_delegate = PlaceDelegate(self.table)
        self.table.setItemDelegateForColumn(0, self.place_delegate)
        self.place_details_delegate = PlaceDetailsDelegate(controller, defines_completer, self.table)
        self.table.setItemDelegateForColumn(1, self.place_details_delegate)

        self.resize(800, 400)

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 6)
        layout.addWidget(table_with_manipulators(self.table))

        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui. QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)


class PlaceDelegate(QtGui.QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        schema = index.model().schema

        opts = PLACES_EDITORS[schema.mesh_type].keys()

        if index.column() == 0:
            self._first_enter = True

        if opts is None:
            ed = QtGui.QLineEdit(parent)
            return ed

        combo = QtGui.QComboBox(parent)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(opts)
        combo.setMaxVisibleItems(len(opts))
        if index.column() == 0:
            try:
                combo.setCurrentIndex(opts.index(index.data()))
            except ValueError:
                combo.setCurrentIndex(0)
        else:
            combo.setEditable(True)
            combo.setEditText(index.data())
            completer = combo.completer()
            completer.setCaseSensitivity(Qt.CaseSensitive)
            combo.setCompleter(completer)

        return combo

    def eventFilter(self, editor, event):
        if isinstance(editor, QtGui.QComboBox) and event.type() == QtCore.QEvent.Enter and self._first_enter:
            editor.showPopup()
            self._first_enter = False
            return True
        else:
            return super(PlaceDelegate, self).eventFilter(editor, event)


class PlaceDetailsDelegate(HTMLDelegate):

    def __init__(self, controller, defines=None, parent=None):
        super(PlaceDetailsDelegate, self).__init__(parent)
        self.controller = controller
        self.defines = defines

    def createEditor(self, parent, option, index):
        schema = index.model().schema
        model = index.model()
        row = index.row()
        place = model.entries[row][0]
        editor = PLACES_EDITORS[schema.mesh_type][place.label](self.controller, self.defines, parent)
        return editor

    def setEditorData(self, editor, index):
        model = index.model()
        row = index.row()
        place = model.entries[row][0]
        editor.load_data(place)

    def setModelData(self, editor, model, index):
        row = index.row()
        place = model.entries[row][0]
        editor.save_data(place)


PLACES_EDITORS = {
    'Rectangular2D': OrderedDict((
        ("Left", RectangularPlaceSide),
        ("Right", RectangularPlaceSide),
        ("Top", RectangularPlaceSide),
        ("Bottom", RectangularPlaceSide),
        ("Horizontal Line", RectangularPlaceLine),
        ("Vertical Line", RectangularPlaceLine),
    )),
    'Rectangular3D': OrderedDict((
        ("Left", RectangularPlaceSide),
        ("Right", RectangularPlaceSide),
        ("Top", RectangularPlaceSide),
        ("Bottom", RectangularPlaceSide),
        ("Front", RectangularPlaceSide),
        ("Back", RectangularPlaceSide),
    )),
}
