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

from ...qt.QtCore import *
from ...qt.QtGui import *
from ...qt.QtWidgets import *
from ...utils.str import none_to_empty, empty_to_none
from ...utils.widgets import HTMLDelegate, ComboBox, table_edit_shortcut
from ..defines import DefinesCompletionDelegate, get_defines_completer
from ..table import table_with_manipulators
from ...model.solvers.bconds import RectangularBC, BoundaryConditionsModel

try:
    from ..geometry.plot_widget import PlotWidget as GeometryPlotWidget, NavigationToolbar as GeometryNavigationToolbar
except ImportError:
    preview_available = False
else:
    preview_available = True


class PlaceDetailsEditor(QWidget):
    pass


class RectangularPlaceSide(PlaceDetailsEditor):
    Model = RectangularBC.PlaceSide

    def __init__(self, controller, defines=None, parent=None):
        super(RectangularPlaceSide, self).__init__(parent)
        self.controller = controller
        self.setAutoFillBackground(True)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 4, 0)
        self.object = ComboBox()
        self.object.sizePolicy().setHorizontalStretch(2)
        self.object.sizePolicy().setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.object.setEditable(True)
        self.path = ComboBox()
        self.path.sizePolicy().setHorizontalStretch(1)
        self.path.sizePolicy().setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.path.setEditable(True)
        if defines is not None:
            self.object.setCompleter(defines)
            self.path.setCompleter(defines)
        label = QLabel(" Obj&ect:")
        label.setBuddy(self.object)
        label.setFixedWidth(label.fontMetrics().width(label.text()))
        layout.addWidget(label)
        layout.addWidget(self.object)
        label = QLabel(" &Path:")
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
        try: self.object.addItems([''] + list(self.controller.document.geometry.model.names()))
        except AttributeError: pass
        self.object.setCurrentIndex(self.object.findText(data.object))
        self.object.setEditText(none_to_empty(data.object))
        self.path.clear()
        try: self.path.addItems([''] + list(self.controller.document.geometry.model.paths()))
        except AttributeError: pass
        self.path.setCurrentIndex(self.object.findText(data.path))
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
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 4, 0)
        self.position = QLineEdit()
        self.start = QLineEdit()
        self.stop = QLineEdit()
        self.position.sizePolicy().setHorizontalStretch(1)
        self.start.sizePolicy().setHorizontalStretch(1)
        self.stop.sizePolicy().setHorizontalStretch(1)
        if defines is not None:
            self.position.setCompleter(defines)
            self.start.setCompleter(defines)
            self.stop.setCompleter(defines)
        label = QLabel(" &Pos:")
        label.setBuddy(self.position)
        label.setFixedWidth(label.fontMetrics().width(label.text()))
        layout.addWidget(label)
        layout.addWidget(self.position)
        label = QLabel(" &From:")
        label.setBuddy(self.start)
        label.setFixedWidth(label.fontMetrics().width(label.text()))
        layout.addWidget(label)
        layout.addWidget(self.start)
        label = QLabel(" &To:")
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


if preview_available:
    class NavigationToolbar(GeometryNavigationToolbar):

        toolitems = (
            ('Plot', 'Plot selected geometry object', 'draw-brush', 'plot', None),
            ('Refresh', 'Refresh plot after each change of geometry', 'view-refresh', 'auto_refresh', True),
            (None, None, None, None, None),
            ('Home', 'Zoom to whole geometry', 'go-home', 'home', None),
            ('Back', 'Back to previous view', 'go-previous', 'back', None),
            ('Forward', 'Forward to next view', 'go-next', 'forward', None),
            (None, None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'transform-move', 'pan', False),
            ('Zoom', 'Zoom to rectangle', 'zoom-in', 'zoom', False),
            (None, None, None, None, None),
            ('Aspect', 'Set equal aspect ratio for both axes', 'system-lock-screen', 'aspect', False),
            (None, None, None, None, None),
            ('Plane:', 'Select longitudinal-transverse plane', None, 'select_plane',
             (('tran-long', 'long-vert', 'tran-vert'), 2)),
        )

        def __init__(self, *args, **kwargs):
            super(NavigationToolbar, self).__init__(*args, **kwargs)
            self._actions['plot'].setShortcut(Qt.ALT + Qt.Key_P)
            self.disable_planes(('long','tran','vert'))

        def home(self):
            if self.controller.plotted_geometry is not None:
                box = self.controller.plotted_geometry.bbox
                self.parent.zoom_bbox(box)

        def select_plane(self, index):
            plane = ('10', '02', '12')[index]
            self._axes = self._axes_names[int(plane[0])], self._axes_names[int(plane[1])]
            self.controller.checked_plane = plane
            if self.controller.plot_auto_refresh: self.controller.plot()
            self.set_message(self.mode)


    class PlotWidget(GeometryPlotWidget):
        pass


class BoundaryConditionsDialog(QDialog):

    def __init__(self, controller, schema, data, parent=None):
        super(BoundaryConditionsDialog, self).__init__(parent)
        self.setWindowTitle(schema.label2 + " Boundary Conditions")

        self.schema = schema
        self.data = data

        self.plotted_geometry = None
        self.plotted_mesh = None
        self.plot_auto_refresh = True
        self.checked_plane = '12'

        self.table = QTableView()
        model = BoundaryConditionsModel(schema, data)
        self.table.setModel(model)
        cols = model.columnCount(None)  # column widths:
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 250)
        try:
            self.table.horizontalHeader().setResizeMode(1, QHeaderView.Stretch)
        except AttributeError:
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        table_edit_shortcut(self.table, 0, QKeySequence(Qt.Key_P))
        table_edit_shortcut(self.table, 1, QKeySequence(Qt.Key_D))
        used_shortcuts = ['p', 'd']

        self.defines_delegate = DefinesCompletionDelegate(controller.document.defines.model, self.table)
        defines_completer = get_defines_completer(controller.document.defines.model, self)
        for i in range(2, model.columnCount()):
            self.table.setColumnWidth(i, 150)
            self.table.setItemDelegateForColumn(i, self.defines_delegate)
            label = schema.keys[i-2].lower()
            for l in label:
                if l not in used_shortcuts:
                    table_edit_shortcut(self.table, i, QKeySequence(l))
                    used_shortcuts.append(l)
                    break

        self.place_delegate = PlaceDelegate(self.table)
        self.table.setItemDelegateForColumn(0, self.place_delegate)
        self.place_details_delegate = PlaceDetailsDelegate(controller, defines_completer, self.table)
        self.table.setItemDelegateForColumn(1, self.place_details_delegate)

        self.resize(800, 400)

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 6)
        layout.addWidget(table_with_manipulators(self.table))

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |  QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def plot_boundaries(self):
        if not preview_available: return


class PlaceDelegate(QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        schema = index.model().schema

        opts = PLACES_EDITORS[schema.mesh_type].keys()

        if index.column() == 0:
            self._first_enter = True

        if opts is None:
            ed = QLineEdit(parent)
            return ed
        else:
            opts = list(opts)

        combo = QComboBox(parent)
        combo.setInsertPolicy(QComboBox.NoInsert)
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
        if isinstance(editor, QComboBox) and event.type() == QEvent.Enter and self._first_enter:
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
