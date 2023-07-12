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

# coding: utf8
import sys
import weakref
from lxml import etree

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from .. import Controller, select_index_from_info
from ...utils import getattr_by_path, setattr_by_path
from ...utils.widgets import table_last_col_fill, table_edit_shortcut
from ...utils.qsignals import BlockQtSignals
from ..table import table_and_manipulators
from ...utils.qundo import UndoCommandWithSetter
from ...model.grids import GridsModel, construct_grid
from ...model.materials import HandleMaterialsModule
from ...model.info import Info
from ...utils import get_manager
from ...utils.xml import XMLparser
from ...utils.config import CONFIG

basestring = str, bytes
try:
    import plask
    from .plot_widget import PlotWidget
except ImportError:
    plask = None
else:
    import plask.mesh
    from ...utils.matplotlib import BwColor

def msize(x):
    return '{:,}'.format(len(x)).replace(',', ' ')

class GridsController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = GridsModel()
        Controller.__init__(self, document, model)

        weakself = weakref.proxy(self)

        self._current_index = None
        self._last_index = None
        self._current_controller = None
        self._geometry_name = None
        self._lims = None

        self.manager = None

        self.splitter = QSplitter()

        self.grids_table = QTableView()
        self.grids_table.setModel(self.model)
        #self.grids_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model, self.grids_table))
        #self.materialsTableActions = TableActions(self.grids_table)
        table_last_col_fill(self.grids_table, self.model.columnCount(None), 80)
        self.grids_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.grids_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table_edit_shortcut(self.grids_table, 0, 'n')
        table, toolbar = table_and_manipulators(self.grids_table, self.splitter, title="Meshes and Generators")
        self.splitter.addWidget(table)
        self.grids_table.setVisible(False)
        try:
            self.grids_table.horizontalHeader().setResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        except AttributeError:
            self.grids_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.grids_table.setVisible(True)

        self.vertical_splitter = QSplitter()
        self.vertical_splitter.setOrientation(Qt.Orientation.Vertical)

        if plask is not None:
            self.mesh_preview = PlotWidget(self, self.vertical_splitter)
            # self.status_bar = QLabel()
            # self.status_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark)")
            # self.mesh_preview.layout().addWidget(self.status_bar)
            self.vertical_splitter.addWidget(self.mesh_preview)
            self.generate_mesh_action = QAction(QIcon.fromTheme('document-save'), "Make Mesh", toolbar)
            CONFIG.set_shortcut(self.generate_mesh_action, 'mesh_make')
            self.generate_mesh_action.triggered.connect(self.generate_mesh)
            self.generate_mesh_action.setEnabled(False)
            toolbar.addSeparator()
            toolbar.addAction(self.generate_mesh_action)
        else:
            self.mesh_preview = None
            self.generate_mesh_action = None

        self.parent_for_editor_widget = QStackedWidget()
        self.vertical_splitter.addWidget(self.parent_for_editor_widget)

        self.splitter.addWidget(self.vertical_splitter)

        focus_action = QAction(self.grids_table)
        focus_action.triggered.connect(lambda: weakself.parent_for_editor_widget.currentWidget().setFocus())
        focus_action.setShortcut(QKeySequence(Qt.Key.Key_Return))
        focus_action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self.grids_table.addAction(focus_action)

        self.splitter.setSizes([10000,26000])

        selection_model = self.grids_table.selectionModel()
        selection_model.selectionChanged.connect(self.grid_selected) #currentChanged ??

        self.model.changed.connect(self.on_model_change)

        self.selected_geometry = None
        self.current_geometry = None
        self.checked_plane = '12'
        self.current_model = self.current_mesh = None
        self.plot_auto_refresh = False
        self.geometry_axes_names = {}

        self.document.window.config_changed.connect(self.reconfig)

    def update_geometries(self):
        if plask is not None:
            try:
                dim = max(self._current_controller.model.dim, 2)
            except AttributeError:
                return
            geoms = [('', ('long','tran','vert'))] + \
                    list((r.name, r.get_axes_conf()) for r in self.document.geometry.model.get_roots(dim=dim)
                         if r.name is not None)
            self.geometry_axes_names = dict(geoms)
            geometry_list = self.mesh_preview.toolbar.widgets['select_geometry']
            with BlockQtSignals(geometry_list):
                geometry_list.clear()
                geometry_list.addItems([g[0] for g in geoms])
            try:
                geometry_name = self._current_controller.model.geometry_name
                geometry_list.setCurrentIndex(geoms.index(geometry_name))
                if dim == 3:
                    self.mesh_preview.toolbar.enable_planes(
                        self.geometry_axes_names.get(geometry_name, ('long','tran','vert')))
                else:
                    self.mesh_preview.toolbar.disable_planes(
                        self.geometry_axes_names.get(geometry_name, ('long','tran','vert')))
            except (AttributeError, ValueError):
                pass

    def get_widget(self):
        return self.splitter

    def save_data_in_model(self):
        if self._current_controller is not None:
            self._current_controller.save_data_in_model()

    def on_edit_enter(self):
        self.grids_table.selectionModel().clear()   # model could have completely changed
        if self._last_index is not None:
            self.grids_table.selectRow(self._last_index)
            self.update_geometries()
            if plask is not None:
                model = self._current_controller.model
                if self._geometry_name in self.geometry_axes_names.keys():
                    with BlockQtSignals(self.mesh_preview.toolbar.widgets['select_geometry']):
                        self.mesh_preview.toolbar.widgets['select_geometry'].setCurrentText(self._geometry_name)
                    model.geometry_name = self._geometry_name
                self.update_current_mesh()
                if self._lims is not None:
                    self.mesh_preview.axes.set_xlim(self._lims[0])
                    self.mesh_preview.axes.set_ylim(self._lims[1])
                if self.current_mesh and self.current_geometry:
                    if self.plot_auto_refresh:
                        self.plot()
                    else:
                        self.show_update_required()
        elif plask is not None:
            self.show_update_required()
        self.grids_table.setFocus()

    def on_edit_exit(self):
        self.manager = None
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.grids_table.selectionModel().clear()
            self._geometry_name = self.mesh_preview.toolbar.widgets['select_geometry'].currentText()
            self._lims = self.mesh_preview.axes.get_xlim(), self.mesh_preview.axes.get_ylim()
        return True

    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.window)

    def select_info(self, info):
        try: action = info.action
        except AttributeError: pass
        else: getattr(self, action)()
        if select_index_from_info(info, self.model, self.grids_table):
            self._current_controller.select_info(info) # try to select property

    def grid_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0].row() if indexes else None)):
            self.grids_table.selectionModel().select(old_selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def set_current_index(self, new_index):
        """
            Try to change current index.
            :param int new_index: index of new current script
            :return: False only when script should restore old selection
        """
        if self._current_index == new_index: return True
        if self._current_controller is not None:
            if not self._current_controller.on_edit_exit():
                self.vertical_splitter.setSizes([100000,0])
                if plask is not None:
                    self.mesh_preview.clear()
                    self.show_update_required()
                return False
        self._current_index = new_index
        for i in reversed(range(self.parent_for_editor_widget.count())):
            widget = self.parent_for_editor_widget.widget(i)
            self.parent_for_editor_widget.removeWidget(widget)
            widget.setParent(None)
        if self._current_index is None:
            self._current_controller = None
        else:
            self._current_controller = self.model.entries[new_index].get_controller(self.document)
            self.parent_for_editor_widget.addWidget(self._current_controller.get_widget())
            self._current_controller.on_edit_enter()
            self.update_geometries()
            if plask is not None:
                self.update_current_mesh()
                if self.current_mesh and self.current_geometry:
                    if self.plot_auto_refresh:
                        self.plot_mesh(set_limits=True, ignore_no_geometry=True)
                    else:
                        self.mesh_preview.clear()
                        self.show_update_required()
                self.generate_mesh_action.setEnabled(self._current_controller.model.is_generator and
                                                     self.current_geometry is not None)
        self.vertical_splitter.setSizes([100000,1])
        return True

    def show_update_required(self):
        if self._current_controller is not None:
            self.model.clear_info_messages()
            self.model.add_info_message("Mesh changed: click here to update the plot", Info.INFO, action='plot')
            self.model.refresh_info()

    def on_model_change(self, *args, **kwargs):
        self.save_data_in_model()
        if plask is not None:
            if self._current_controller is not None:
                self.update_current_mesh()
                if self.plot_auto_refresh:
                    self.plot_mesh(set_limits=False, ignore_no_geometry=True)
                else:
                    self.show_update_required()

    def update_current_mesh(self, model=None):
        geometry_changed = False
        if model is None:
            model = self._current_controller.model
        self.model.clear_info_messages()
        self.mesh_preview.clear()
        if plask is None: return
        if model is not None:
            geometry_name = self.mesh_preview.toolbar.widgets['select_geometry'].currentText()
            if getattr(model, 'geometry_name', None) in self.geometry_axes_names.keys():
                if geometry_name != model.geometry_name:
                    geometry_name = model.geometry_name
                    with BlockQtSignals(self.mesh_preview.toolbar.widgets['select_geometry']):
                        self.mesh_preview.toolbar.widgets['select_geometry'].setCurrentText(geometry_name)
            else:
                if getattr(model, 'geometry_name', None) != geometry_name:
                    model.geometry_name = geometry_name
                    geometry_changed = True
            if geometry_name is None:
                geometry_changed = True
        try:
            if self.manager is None:
                manager = get_manager()
                with HandleMaterialsModule(self.document):
                    manager.load(self.document.get_contents(sections=('defines', 'materials', 'geometry')))
            else:
                manager = self.manager
                manager.msh.clear()
            manager.load(self.document.get_contents(sections=('grids',)))
            self.manager = manager
            try:
                self.selected_geometry = str(model.geometry_name)
                if self.selected_geometry:
                    self.current_geometry = manager.geo[self.selected_geometry]
                else:
                    self.current_geometry = None
            except KeyError:
                self.current_geometry = None
                self.selected_geometry = None
            if model.is_mesh:
                mesh = manager.msh[model.name]
            elif model.is_generator:
                if self.current_geometry is None:
                    self.generate_mesh_action.setEnabled(False)
                    mesh = None
                else:
                    self.generate_mesh_action.setEnabled(True)
                    mesh = manager.msh[model.name](self.current_geometry)
            else:
                mesh = None
            self.need_reset_plot = model != self.current_model or geometry_changed

        except Exception as e:
            # self.model.add_info_message("Could not update mesh preview: {}".format(str(e)), Info.ERROR)
            self.model.add_info_message(str(e), Info.ERROR)
            from ... import _DEBUG
            if _DEBUG:
                import traceback
                traceback.print_exc()
            self.mesh_preview.info.setVisible(False)
            self.current_mesh = None
            res = False

        else:
            self.current_model = model
            self.current_mesh = mesh

            if mesh is None:
                self.mesh_preview.info.setVisible(False)
            else:
                self.mesh_preview.info.setVisible(True)
                info = f"  Mesh size:   {msize(mesh)}  nodes"

                if isinstance(mesh, plask.mesh.Rectangular2D):
                    info += f"   [{msize(mesh.axis0)} x {msize(mesh.axis1)}]    "
                elif isinstance(mesh, plask.mesh.Rectangular3D):
                    info += f"   [{msize(mesh.axis0)} x {msize(mesh.axis1)} x {msize(mesh.axis2)}]   "
                else:
                    info += ",   "

                try:
                    info += f"{msize(mesh.elements)} elements"
                except AttributeError:
                    pass
                self.mesh_preview.info.setText(info)

            res = True

        for err in manager.errors:
            self.model.add_info_message(err, Info.WARNING)
        self.model.refresh_info()

        return res

    def plot_mesh(self, set_limits, ignore_no_geometry=False):
        if self.need_reset_plot:
            try:
                self.mesh_preview.toolbar._nav_stack.clear()
            except AttributeError:
                self.mesh_preview.toolbar._views.clear()
            self.need_reset_plot = False

        if self.current_geometry is None:
            if ignore_no_geometry:
                return
            else:
                raise ValueError("You must select geometry to preview generators")

        self.mesh_preview.update_plot(self.current_mesh, self.current_geometry, set_limits=set_limits,
                                        plane=self.checked_plane)

        self.model.clear_info_messages()
        self.model.refresh_info()

    def plot(self):
        if self._current_controller is not None:
            if self.current_mesh is None or self.current_geometry is None:
                self.update_current_mesh()
            self.plot_mesh(set_limits=self.need_reset_plot)

    def reconfig(self):
        if plask is not None:
            colors = CONFIG['geometry/material_colors'].copy()
            self.mesh_preview.get_color = BwColor(colors, self.mesh_preview.axes)
            self.mesh_preview.axes.set_facecolor(CONFIG['plots/face_color'])
            self.mesh_preview.axes.grid(True, color=CONFIG['plots/grid_color'])
            if self._current_controller is not None and \
               (self.plot_auto_refresh or hasattr(self._current_controller.model, 'geometry_name')):
                if self.current_mesh is None or self.current_geometry is None:
                    self.update_current_mesh()
                self.plot_mesh(set_limits=True, ignore_no_geometry=True)
            else:
                self.mesh_preview.canvas.draw()


    def generate_mesh(self):
        from ... import _DEBUG
        if plask is None: return
        try:
            if self.manager is None:
                manager = get_manager()
                with HandleMaterialsModule(self.document):
                    manager.load(self.document.get_contents(sections=('defines', 'materials', 'geometry')))
            else:
                manager = self.manager
                manager.msh.clear()
            manager.load(self.document.get_contents(sections=('grids',)))
            self.manager = manager
            mesh = manager.msh[self._current_controller.model.name](manager.geo[self.selected_geometry])
            name = self._current_controller.model.name + '-' + self.selected_geometry
            xml = plask.XmlWriter({}, {name: mesh}, {})
            if _DEBUG:
                print(xml, file=sys.stderr)
            mesh_model = construct_grid(self.model,
                                        etree.parse(StringIO(str(xml)), XMLparser).getroot().find('grids/mesh'))
            mesh_model.geometry_name = self.selected_geometry
            index = self.grids_table.selectionModel().currentIndex()
            if index.isValid():
                row = self.model.insert(index.row()+1, mesh_model)
            else:
                row = self.model.insert(len(self.model.entries), mesh_model)
            if row is not None: self.grids_table.selectRow(row)

        except Exception as e:
            QMessageBox.critical(None, "Mesh Generation Error",
                                 "Could not generate mesh with specified generator:\n\n{}".format(str(e)))
            if _DEBUG:
                import traceback
                traceback.print_exc()


class GridController(Controller):

    def __init__(self, document=None, model=None):
        super().__init__(document, model)
        #self.grid_model.changed.connect(self._model_change_cb)

    @property
    def grid_model(self):
        """ :return Grid: model of edited grid """
        return self.model

    @property
    def section_model(self):
        """ :return GridsModel: model of a whole grids's section """
        return self.grid_model.tree_parent

    def _change(self, setter, value, old_value, label):
        if value != old_value:
            grid_model = self.grid_model
            self.section_model.undo_stack.push(UndoCommandWithSetter(
                self.section_model, lambda v: setter(grid_model, v),
                value, old_value, "change grid's {}".format(label)
            ))

    def _change_attr(self, attr, value, label=None):
        old_value = getattr_by_path(self.grid_model, attr, default=None)
        if value != old_value:
            if label is None:
                label = attr
                while not isinstance(label, basestring):
                    label = label[0]
            grid_model = self.grid_model
            self.section_model.undo_stack.push(UndoCommandWithSetter(
                self.section_model, lambda v: setattr_by_path(grid_model, attr, v),
                value, old_value, "change grid's {}".format(label)
            ))

    def fill_form(self):
        pass

    def _fill_form_cb(self, *args, **kwargs):
        self.fill_form()

    def on_edit_enter(self):
        super().on_edit_enter()
        self.section_model.changed.connect(self._fill_form_cb)
        self.fill_form()

    def on_edit_exit(self):
        self.manager = None
        self.section_model.changed.disconnect(self._fill_form_cb)
        return super().on_edit_exit()

    def select_info(self, info):
        try:
            getattr_by_path(self, info.property).setFocus()
        except: pass
