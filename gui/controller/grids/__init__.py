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

from ...qt.QtCore import Qt
from ...qt.QtGui import QSplitter

from ...qt import QtGui
from .. import Controller, select_index_from_info
from ...utils import getattr_by_path, setattr_by_path
from ...utils.widgets import table_last_col_fill, table_edit_shortcut
from ...utils.qsignals import BlockQtSignals
from ..table import table_with_manipulators
from ...utils.qundo import UndoCommandWithSetter
from ...model.grids import GridsModel
from ...model.info import Info

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str


try:
    import plask
    from .plot_widget import PlotWidget
except ImportError:
    plask = None


class GridsController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = GridsModel()
        Controller.__init__(self, document, model)

        self._current_index = None
        self._last_index = None
        self._current_controller = None

        self.splitter = QSplitter()

        self.grids_table = QtGui.QTableView()
        self.grids_table.setModel(self.model)
        #self.grids_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model, self.grids_table))
        #self.materialsTableActions = TableActions(self.grids_table)
        table_last_col_fill(self.grids_table, self.model.columnCount(None), 80)
        self.grids_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.grids_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        table_edit_shortcut(self.grids_table, 0, 'n')
        self.splitter.addWidget(table_with_manipulators(self.grids_table, self.splitter, title="Meshes and Generators"))
        self.grids_table.setVisible(False)
        self.grids_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.grids_table.setVisible(True)

        self.vertical_splitter = QtGui.QSplitter()
        self.vertical_splitter.setOrientation(Qt.Vertical)

        if plask is not None:
            self.mesh_preview = PlotWidget(self, self.vertical_splitter)
            # self.status_bar = QtGui.QLabel()
            # self.status_bar.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark)")
            # self.mesh_preview.layout().addWidget(self.status_bar)
            self.vertical_splitter.addWidget(self.mesh_preview)

        else:
            self.mesh_preview = None

        self.parent_for_editor_widget = QtGui.QStackedWidget()
        self.vertical_splitter.addWidget(self.parent_for_editor_widget)

        self.splitter.addWidget(self.vertical_splitter)

        focus_action = QtGui.QAction(self.grids_table)
        focus_action.triggered.connect(lambda: self.parent_for_editor_widget.currentWidget().setFocus())
        focus_action.setShortcut(QtGui.QKeySequence(Qt.Key_Return))
        focus_action.setShortcutContext(Qt.WidgetShortcut)
        self.grids_table.addAction(focus_action)

        self.splitter.setSizes([10000,26000])

        selection_model = self.grids_table.selectionModel()
        selection_model.selectionChanged.connect(self.grid_selected) #currentChanged ??

        self.model.changed.connect(self.on_model_change)

        self.checked_plane = '12'
        self.plotted_model = self.plotted_mesh = None
        self.plot_auto_refresh = False

    def update_geometries(self):
        if plask is not None:
            dim = max(self._current_controller.model.dim, 2)
            if dim == 3:
                self.mesh_preview.toolbar.enable_planes(('long','tran','vert'))
            else:
                self.mesh_preview.toolbar.disable_planes(('long','tran','vert'))
            geoms = [''] + list(r.name for r in self.document.geometry.model.get_roots(dim=dim) if r.name is not None)
            geometry_list = self.mesh_preview.toolbar.widgets['select_geometry']
            with BlockQtSignals(geometry_list):
                geometry_list.clear()
                geometry_list.addItems(geoms)
            try:
                geometry_list.setCurrentIndex(geoms.index(self._current_controller.model.geometry_name))
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
            if plask is not None and self.plot_auto_refresh:
                self.plot()
        elif plask is not None:
            self.show_update_required()
        self.grids_table.setFocus()

    def on_edit_exit(self):
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.grids_table.selectionModel().clear()
        return True

    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.window)

    def select_info(self, info):
        try: action = info.action
        except AttributeError: pass
        else: return action()
        if select_index_from_info(info, self.model, self.grids_table):
            self._current_controller.select_info(info) # try to select property

    def grid_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0].row() if indexes else None)):
            self.grids_table.selectionModel().select(old_selection, QtGui.QItemSelectionModel.ClearAndSelect)

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
            self.parent_for_editor_widget.removeWidget(self.parent_for_editor_widget.widget(i))
        if self._current_index is None:
            self._current_controller = None
        else:
            self._current_controller = self.model.entries[new_index].get_controller(self.document)
            self.parent_for_editor_widget.addWidget(self._current_controller.get_widget())
            self._current_controller.on_edit_enter()
            self.update_geometries()
            if plask is not None:
                if self.plot_auto_refresh or hasattr(self._current_controller.model, 'geometry_name'):
                    self.plot_mesh(self._current_controller.model, set_limits=True, ignore_no_geometry=True)
                else:
                    self.mesh_preview.clear()
                    self.show_update_required()
        self.vertical_splitter.setSizes([100000,1])
        return True

    def show_update_required(self):
        if self._current_controller is not None:
            self.model.info_message("Mesh changed: click here to update the plot", Info.INFO, action=self.plot)
            # self.status_bar.setText("Press Alt+P to update the plot")
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark); background-color: #ffff88;")
        else:
            self.model.info_message()
            # self.status_bar.setText('')
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark); background-color: palette(background);")

    def on_model_change(self, *args, **kwargs):
        self.save_data_in_model()
        if plask is not None:
            if self.plot_auto_refresh:
                if self._current_controller is not None:
                    self.plot_mesh(self._current_controller.model, set_limits=False, ignore_no_geometry=True)
            else:
                self.show_update_required()

    def plot_mesh(self, model, set_limits, ignore_no_geometry=False):
        self.mesh_preview.clear()
        if model is not None:
            model.geometry_name = self.mesh_preview.toolbar.widgets['select_geometry'].currentText()
        if plask is None:
            return
        manager = plask.Manager(draft=True)
        try:
            manager.load(self.document.get_content(sections=('defines', 'geometry', 'grids')))
            try:
                selected_geometry = str(self.mesh_preview.toolbar.widgets['select_geometry'].currentText())
                if selected_geometry:
                    geometry = manager.geometry[selected_geometry]
                else:
                    geometry = None
            except KeyError:
                geometry = None
            if model.is_mesh:
                mesh = manager.mesh[model.name]
            elif model.is_generator:
                if geometry is None:
                    if ignore_no_geometry:
                        mesh = None
                    else:
                        raise ValueError("You must select geometry to preview generators")
                else:
                    mesh = manager.meshgen[model.name](geometry)
            else:
                mesh = None
            if model != self.plotted_model:
                self.clear = self.mesh_preview.toolbar._views.clear()
            self.mesh_preview.update_mesh_plot(mesh, geometry, set_limits=set_limits, plane=self.checked_plane)
        except Exception as e:
            self.model.info_message("Could not update mesh preview: {}".format(str(e)), Info.WARNING)
            # self.status_bar.setText(str(e))
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark); background-color: #ff8888;")
            # self.status_bar.setAutoFillBackground(True)
            from ... import _DEBUG
            if _DEBUG:
                import traceback
                traceback.print_exc()
            return False
        else:
            self.manager = manager
            self.plotted_model = model
            self.plotted_mesh = mesh
            # if mesh.dim == 3:
            #     self.preview.toolbar.enable_planes(tree_element.get_axes_conf())
            # else:
            #     self.preview.toolbar.disable_planes(tree_element.get_axes_conf())
            self.model.info_message()
            # self.status_bar.setText('')
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark); background-color: palette(background);")
            return True

    def plot(self):
        if self._current_controller is not None:
            self.plot_mesh(self._current_controller.model, set_limits=True)


class GridController(Controller):

    def __init__(self, document=None, model=None):
        super(GridController, self).__init__(document, model)
        #self.grid_model.changed.connect(self._model_change_cb)

    """
        :return Grid: model of edited grid
    """
    @property
    def grid_model(self):
        return self.model

    """
        :return GridsModel: model of a whole grids's section
    """
    @property
    def section_model(self):
        return self.grid_model.tree_parent

    def _change(self, setter, value, old_value, label):
        if value != old_value:
            self.section_model.undo_stack.push(UndoCommandWithSetter(
                self.section_model, lambda v: setter(self.grid_model, v),
                value, old_value, "change grid's {}".format(label)
            ))

    def _change_attr(self, attr, value, label=None):
        old_value = getattr_by_path(self.grid_model, attr, default=None)
        if value != old_value:
            if label is None:
                label = attr
                while not isinstance(label, basestring): label = label[0]
            self.section_model.undo_stack.push(UndoCommandWithSetter(
                self.section_model, lambda v: setattr_by_path(self.grid_model, attr, v),
                value, old_value, "change grid's {}".format(label)
            ))

    def fill_form(self):
        pass

    def _fill_form_cb(self, *args, **kwargs):
        self.fill_form()

    def on_edit_enter(self):
        super(GridController, self).on_edit_enter()
        self.section_model.changed.connect(self._fill_form_cb)
        self.fill_form()

    def on_edit_exit(self):
        self.section_model.changed.disconnect(self._fill_form_cb)
        return super(GridController, self).on_edit_exit()

    def select_info(self, info):
        try:
            getattr_by_path(self, info.property).setFocus()
        except: pass
