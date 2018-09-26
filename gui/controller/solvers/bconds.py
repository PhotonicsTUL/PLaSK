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
import sys
from collections import OrderedDict
from copy import copy

from lxml.etree import tostring

from ...qt.QtCore import *
from ...qt.QtGui import *
from ...qt.QtWidgets import *
from ...qt import QtSignal, QtSlot
from ...utils import get_manager
from ...utils.config import CONFIG
from ...utils.str import none_to_empty, empty_to_none
from ...utils.widgets import HTMLDelegate, ComboBox, table_edit_shortcut
from ...utils.qsignals import BlockQtSignals
from ..defines import DefinesCompletionDelegate, get_defines_completer
from ..table import table_with_manipulators
from ...model.solvers.bconds import RectangularBC, BoundaryConditionsModel

try:
    import plask
    from ...utils.matplotlib import PlotWidgetBase, BwColor
except ImportError:
    preview_available = False
else:
    preview_available = True
    import matplotlib


class PlaceDetailsEditor(QWidget):

    def __init__(self, delegate, parent=None):
        super(PlaceDetailsEditor, self).__init__(parent)
        self.delegate = delegate

    def data_changed(self, *args):
        self.delegate.commitData.emit(self)


class RectangularPlaceSide(PlaceDetailsEditor):
    Model = RectangularBC.PlaceSide

    def __init__(self, delegate, parent=None):
        super(RectangularPlaceSide, self).__init__(delegate, parent)
        self.setAutoFillBackground(True)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 4, 0)
        self.object = ComboBox()
        self.object.sizePolicy().setHorizontalStretch(2)
        self.object.sizePolicy().setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.object.setEditable(True)
        self.object.currentIndexChanged.connect(self.data_changed)
        self.object.editingFinished.connect(self.data_changed)
        self.path = ComboBox()
        self.path.sizePolicy().setHorizontalStretch(1)
        self.path.sizePolicy().setHorizontalPolicy(QSizePolicy.MinimumExpanding)
        self.path.setEditable(True)
        self.path.currentIndexChanged.connect(self.data_changed)
        self.path.editingFinished.connect(self.data_changed)
        if delegate.defines is not None:
            self.object.setCompleter(delegate.defines)
            self.path.setCompleter(delegate.defines)
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

    def fill_details(self, obj, pth):
        with BlockQtSignals(self.object):
            self.object.setCurrentIndex(self.object.findText(obj))
            self.object.setEditText(none_to_empty(obj))
        with BlockQtSignals(self.path):
            self.path.setCurrentIndex(self.path.findText(pth))
            self.path.setEditText(none_to_empty(pth))

    def load_data(self, data):
        self.object.clear()
        try: self.object.addItems([''] + list(self.delegate.controller.document.geometry.model.get_names()))
        except AttributeError: pass
        try: self.path.addItems([''] + list(self.delegate.controller.document.geometry.model.get_paths()))
        except AttributeError: pass
        self.fill_details(data.object, data.path)

    def save_data(self, data):
        data.object = empty_to_none(self.object.currentText())
        data.path = empty_to_none(self.path.currentText())


class RectangularPlaceLine(PlaceDetailsEditor):
    Model = RectangularBC.PlaceLine

    def __init__(self, delegate, parent=None):
        super(RectangularPlaceLine, self).__init__(delegate, parent)
        self.setAutoFillBackground(True)
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 0, 4, 0)
        self.position = QLineEdit()
        self.start = QLineEdit()
        self.stop = QLineEdit()
        self.position.editingFinished.connect(self.data_changed)
        self.start.editingFinished.connect(self.data_changed)
        self.stop.editingFinished.connect(self.data_changed)
        self.position.sizePolicy().setHorizontalStretch(1)
        self.start.sizePolicy().setHorizontalStretch(1)
        self.stop.sizePolicy().setHorizontalStretch(1)
        if delegate.defines is not None:
            self.position.setCompleter(delegate.defines)
            self.start.setCompleter(delegate.defines)
            self.stop.setCompleter(delegate.defines)
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

    class PlotWidget(PlotWidgetBase):

        class NavigationToolbar(PlotWidgetBase.NavigationToolbar):

            def __init__(self, *args, **kwargs):
                super(PlotWidget.NavigationToolbar, self).__init__(*args, **kwargs)
                self._actions['plot'].setShortcut(Qt.ALT + Qt.Key_P)

            def select_plane(self, index):
                super(PlotWidget.NavigationToolbar, self).select_plane(index)
                if self.controller.plot_auto_refresh: self.controller.plot_bboundaries()

        def __init__(self, controller=None, parent=None, picker=None):
            super(PlotWidget, self).__init__(controller, parent)
            self.get_color = BwColor(self.axes)
            self.first = True
            self.picker = picker

        def update_plot(self, bconds, mesh, geometry, colors, plane='12'):
            # for b in bconds:
            #     print list(b.place(mesh, geometry)), b.value
            updater = self.plot_updater(self.first, plane)
            if self.first:
                try:
                    self.toolbar._nav_stack.clear()
                except AttributeError:
                    self.toolbar._views.clear()
            points = []
            if bconds is not None:
                for m in updater:
                    points = plask.plot_boundary(bconds, mesh, geometry, colors=colors, plane=plane, axes=self.axes)
                    if geometry is not None:
                        try:
                            plask.plot_geometry(axes=self.axes, geometry=geometry, fill=True, zorder=1,
                                                plane=plane, lw=1.0, get_color=self.get_color,
                                                margin=m if self.first else None,
                                                picker=self.picker)
                        except:
                            pass
                self.first = False
            self.canvas.draw()
            return points

    class fake_plask_gui_solver(object):
        pass

    sys.modules['fake_plask_gui_solver'] = fake_plask_gui_solver


class BoundaryConditionsDialog(QDialog):

    def __init__(self, controller, schema, data, parent=None):
        super(BoundaryConditionsDialog, self).__init__(parent)
        self.setWindowTitle(schema.label2 + " Boundary Conditions")
        self.setWindowFlags(Qt.Window)

        self.schema = schema
        self.data = data

        self.plotted_geometry = None
        self.plotted_mesh = None
        self.plot_auto_refresh = True
        self.checked_plane = '12'

        self.table = QTableView()
        self.model = BoundaryConditionsModel(schema, data)
        self.table.setModel(self.model)
        cols = self.model.columnCount(None)  # column widths:
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 250)
        self.table.verticalHeader().show()

        try:
            self.table.horizontalHeader().setResizeMode(1, QHeaderView.Stretch)
        except AttributeError:
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        table_edit_shortcut(self.table, 0, QKeySequence(Qt.Key_P))
        table_edit_shortcut(self.table, 1, QKeySequence(Qt.Key_D))
        used_shortcuts = ['p', 'd']

        self.defines_delegate = DefinesCompletionDelegate(controller.document.defines.model, self.table)
        defines_completer = get_defines_completer(controller.document.defines.model, self)
        for i in range(2, self.model.columnCount()):
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
        self.place_details_delegate = PlaceDetailsDelegate(self, controller, defines_completer, self.table)
        self.table.setItemDelegateForColumn(1, self.place_details_delegate)
        self._active_place_editor = None

        self.place_delegate.placeChanged.connect(self.update)

        self.resize(int(0.85 * controller.document.window.width()), int(0.85 * controller.document.window.height()))

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 6)

        if preview_available:
            self.document = controller.document

            self.info = QTextEdit(self)
            self.info.setVisible(False)
            self.info.setReadOnly(True)
            self.info.setContentsMargins(0, 0, 0, 0)
            self.info.setFrameStyle(0)
            pal = self.info.palette()
            pal.setColor(QPalette.Base, QColor("#ffc"))
            self.info.setPalette(pal)
            self.info.acceptRichText()
            self.info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.info.hide()

            try:
                mesh_name = controller.model.mesh if schema.mesh_attr is None else \
                            controller.model.data[schema.mesh_attr['tag']][schema.mesh_attr['attr']]
            except KeyError:
                mesh_name = None
            try:
                geometry_name = controller.model.geometry if schema.geometry_attr is None else \
                                controller.model.data[schema.geometry_attr['tag']][schema.geometry_attr['attr']]
            except KeyError:
                geometry_name = None
            else:
                self.geometry_node = controller.document.geometry.model.find_by_name(geometry_name)
            self.manager = get_manager()
            try:
                self.manager.load(self.document.get_content(sections=('defines', 'geometry', 'grids')))
            except Exception as err:
                self.preview = None
                self.info.setText(str(err))
                self.info.show()
                layout.addWidget(self.info)
                layout.addWidget(table_with_manipulators(self.table))
            else:
                try:
                    self.geometry = self.manager.geo[str(geometry_name)]
                except KeyError:
                    self.geometry = None
                try:
                    mesh = self.manager.msh[str(mesh_name)]
                    if isinstance(mesh, plask.mesh.MeshGenerator):
                        if self.geometry is not None:
                            self.mesh = mesh(self.geometry)
                        else:
                            self.mesh = None
                    else:
                        self.mesh = mesh
                except KeyError:
                    self.mesh = None

                if self.geometry is None or self.mesh is None:
                    self.preview = self.info = None
                    label = QLabel("Specify proper geometry and mesh for the solver to show boundary conditions preview.")
                    layout.addWidget(label)
                    layout.addWidget(table_with_manipulators(self.table))
                else:
                    splitter = QSplitter(self)
                    splitter.setOrientation(Qt.Vertical)
                    self.preview = PlotWidget(self, splitter, picker=True)
                    wid = QWidget()
                    lay = QVBoxLayout()
                    lay.setContentsMargins(0, 0, 0, 0)
                    lay.setSpacing(0)
                    lay.addWidget(self.preview)
                    lay.addWidget(self.info)
                    wid.setLayout(lay)
                    splitter.addWidget(wid)
                    splitter.addWidget(table_with_manipulators(self.table))
                    layout.addWidget(splitter)
                    self.preview.canvas.mpl_connect('pick_event', self.on_pick_object)

            if schema.mesh_type not in fake_plask_gui_solver.__dict__:
                Mesh = getattr(plask.mesh, schema.mesh_type)
                class Solver(plask.Solver):
                    __name__ = schema.mesh_type
                    def load_xpl(self, xpl, manager):
                        self.bconds = Mesh.BoundaryConditions()
                        for tag in xpl:
                            self.bconds.read_from_xpl(tag, manager)
                setattr(fake_plask_gui_solver, schema.mesh_type, Solver)
            if self.preview is not None:
                self.model.dataChanged.connect(self.update_plot)
                self.model.rowsInserted.connect(self.update_plot)
                self.model.rowsRemoved.connect(self.update_plot)
                self.model.rowsMoved.connect(self.update_plot)
                selection_model = self.table.selectionModel()
                selection_model.selectionChanged.connect(self.highlight_plot)
        else:
            self.preview = self.info = None
            layout.addWidget(table_with_manipulators(self.table))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

        self._picked_path = None
        self.points = []

        self.plot()

    def on_pick_object(self, event):
        # This seems as an ugly hack, but in reality this is the only way to make sure
        # that `setCurrentIndex` is called only once if there are multiple artists in
        # the clicked spot.
        self._picked_path = event.artist.plask_real_path
        QMetaObject.invokeMethod(self, '_picked_object', Qt.QueuedConnection)

    @QtSlot()
    def _picked_object(self):
        if self._picked_path is None: return
        _picked_path = self._picked_path
        self._picked_path = None
        if self._active_place_editor is not None and self.geometry_node is not None:
            while _picked_path:
                node = self.geometry_node.get_node_by_real_path(_picked_path)
                if node.name:
                    node_path = node.path.split(',')[0].strip() if node.path is not None else None
                    self._active_place_editor.fill_details(node.name, node_path)
                    self.place_details_delegate.commitData.emit(self._active_place_editor)
                    break
                else:
                    _picked_path.pop()

    def message(self, msg):
        if msg:
            self.info.setText(msg)
            self.info.show()
            self.info.setFixedHeight(self.info.document().size().height())
        else:
            self.info.clear()
            self.info.hide()

    def highlight_plot(self):
        if self.points:
            colors = [CONFIG['boundary_conditions/color']] * len(self.points)
            for index in self.table.selectionModel().selectedRows():
                colors[index.row()] = CONFIG['boundary_conditions/selected_color']
            for points, color in zip(self.points, colors):
                points.set_color(color)
        self.preview.canvas.draw()

    def showEvent(self, event):
        super(BoundaryConditionsDialog, self).showEvent(event)
        if self.info is not None and self.info.isVisible():
            self.info.setFixedHeight(self.info.document().size().height())

    def resizeEvent(self, event):
        if self.info is not None and self.info.isVisible():
            self.info.setFixedHeight(self.info.document().size().height())

    def update_plot(self, index1=None, index2=None):
        if self.plot_auto_refresh and (index1 is None or index1.column() < 2):
            self.plot()

    def plot(self):
        from ... import _DEBUG
        if not preview_available or self.preview is None or \
           self.geometry is None or self.mesh is None:
            return

        text = self.document.get_content(sections=('defines'))
        xml = self.schema.to_xml(self.model.entries)
        xml = tostring(xml, encoding='utf8').decode('utf8') if xml is not None else ''
        if _DEBUG:
            print(xml)
        text = text[:-9] + """\
          <solvers>
            <local name="bconds" solver="{solver}" lib="fake_plask_gui_solver">
                {value}
            </local>
          </solvers>
        </plask>""".format(solver=self.schema.mesh_type, value=xml)
        try:
            for p in self.points:
                p.remove()
            self.points = []
            self.preview.clear()
            self.manager.solvers.clear()
            self.manager.load(text)
            solver = self.manager.solvers['bconds']
            plotted_bconds = solver.bconds
            colors = [CONFIG['boundary_conditions/color']] * len(plotted_bconds)
            for index in self.table.selectionModel().selectedRows():
                colors[index.row()] = CONFIG['boundary_conditions/selected_color']

            self.points = self.preview.update_plot(plotted_bconds, self.mesh, self.geometry,
                                                   plane=self.checked_plane, colors=colors)
            self.plotted_bconds = plotted_bconds
            self.message(None)
        except Exception as e:
            self.message(str(e))
            self.plotted_bconds = None
            if _DEBUG:
                import traceback
                traceback.print_exc()


class PlaceDelegate(QStyledItemDelegate):

    placeChanged = QtSignal()

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

        combo.currentIndexChanged.connect(lambda: self.placeChanged.emit())

        return combo

    def eventFilter(self, editor, event):
        if isinstance(editor, QComboBox) and event.type() == QEvent.Enter and self._first_enter:
            editor.showPopup()
            self._first_enter = False
            return True
        else:
            return super(PlaceDelegate, self).eventFilter(editor, event)


class PlaceDetailsDelegate(HTMLDelegate):

    def __init__(self, dialog, controller, defines=None, parent=None):
        super(PlaceDetailsDelegate, self).__init__(parent)
        self.dialog = dialog
        self.controller = controller
        self.defines = defines
        self.closeEditor.connect(self.on_close_editor)

    def createEditor(self, parent, option, index):
        schema = index.model().schema
        model = index.model()
        row = index.row()
        place = model.entries[row][0]
        editor = PLACES_EDITORS[schema.mesh_type][place.label](self, parent)
        self.dialog._active_place_editor = editor
        return editor

    def on_close_editor(self, editor):
        self.dialog._active_place_editor = None

    def setEditorData(self, editor, index):
        model = index.model()
        row = index.row()
        place = model.entries[row][0]
        editor.load_data(place)

    def setModelData(self, editor, model, index):
        row = index.row()
        place = model.entries[row][0]
        editor.save_data(place)
        model.dataChanged.emit(index, index)


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
