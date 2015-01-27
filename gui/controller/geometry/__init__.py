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
import operator

import plask

from ...qt import QtGui, QtCore
from ...model.geometry import GeometryModel
from ...model.geometry.constructor import construct_by_name, construct_using_constructor
from ...model.geometry.types import geometry_types_geometries_core, gname

from .. import Controller
from ...utils.widgets import HTMLDelegate

from .plot_widget import PlotWidget


# TODO use ControllerWithSubController (?)
class GeometryController(Controller):

    def _add_child(self, type_constructor, parent_index):
        pos = len(parent_index.internalPointer().children)
        self.model.beginInsertRows(parent_index, pos, pos)
        construct_using_constructor(type_constructor, parent_index.internalPointer())
        self.model.endInsertRows()
        self.tree.setExpanded(parent_index, True)
        new_index = self.model.index(pos, 0, parent_index)
        self.tree.selectionModel().select(new_index,
                                          QtGui.QItemSelectionModel.Clear | QtGui.QItemSelectionModel.Select |
                                          QtGui.QItemSelectionModel.Rows)
        self.tree.setCurrentIndex(new_index)
        self.update_actions()

    def _get_add_child_menu(self, geometry_node_index):
        geometry_node = geometry_node_index.internalPointer()
        if geometry_node is None or not geometry_node.accept_new_child(): return None
        first = True
        result = QtGui.QMenu()
        for section in geometry_node.add_child_options():
            if not first:
                result.addSeparator()
            first = False
            for type_name, type_constructor in sorted(section.items(), key=operator.itemgetter(0)):
                if type_name.endswith('2d') or type_name.endswith('3d'):
                    type_name = type_name[:-2]
                a = QtGui.QAction(gname(type_name), result)
                a.triggered[()].connect(lambda type_constructor=type_constructor, parent_index=geometry_node_index:
                                        self._add_child(type_constructor, parent_index))
                result.addAction(a)
        return result

    def fill_add_menu(self):
        self.add_menu.clear()
        current_index = self.tree.selectionModel().currentIndex()
        if current_index.isValid():
            add_child_menu = self._get_add_child_menu(current_index)
            if add_child_menu:
                self.add_menu.addAction('Item').setMenu(add_child_menu)
        for n in geometry_types_geometries_core.keys():
            a = QtGui.QAction(gname(n), self.add_menu)
            a.triggered[()].connect(lambda n=n: self.append_geometry_node(n))
            self.add_menu.addAction(a)

    def update_actions(self):
        has_selected_object = not self.tree.selectionModel().selection().isEmpty()
        self.remove_action.setEnabled(has_selected_object)
        self.plot_action.setEnabled(has_selected_object)
        self.fill_add_menu()

        u, d = self.model.can_move_node_up_down(self.tree.selectionModel().currentIndex())
        self.move_up_action.setEnabled(u)
        self.move_down_action.setEnabled(d)

        #hasCurrent = self.tree.selectionModel().currentIndex().isValid()
        #self.insertRowAction.setEnabled(hasCurrent)
        #self.insertColumnAction.setEnabled(hasCurrent)

    def append_geometry_node(self, type_name):
        self.tree.model().append_geometry(type_name)
        new_index = self.model.index(len(self.tree.model().roots)-1, 0)
        self.tree.selectionModel().select(new_index,
                                          QtGui.QItemSelectionModel.Clear | QtGui.QItemSelectionModel.Select |
                                          QtGui.QItemSelectionModel.Rows)
        self.tree.setCurrentIndex(new_index)

    def remove_node(self):
        index = self.tree.selectionModel().currentIndex()
        model = self.tree.model()
        if (model.removeRow(index.row(), index.parent())):
            self.update_actions()

    def _swap_neighbour_nodes(self, parent_index, row1, row2):
        if self.model.is_read_only(): return
        if row2 < row1: row1, row2 = row2, row1
        children = self.model.children_list(parent_index)
        if row1 < 0 or row2 < len(children): return
        self.model.beginMoveRows(parent_index, row2, row2, parent_index, row1)
        children[row1], children[row2] = children[row2], children[row1]
        self.model.endMoveRows()
        self.fire_changed()

    def move_current_up(self):
        self.model.move_node_up(self.tree.selectionModel().currentIndex())
        self.update_actions()

    def move_current_down(self):
        self.model.move_node_down(self.tree.selectionModel().currentIndex())
        self.update_actions()

    def plot(self):
        current_index = self.tree.selectionModel().currentIndex()
        if not current_index.isValid(): return
        tree_element = current_index.internalPointer()

        #TODO support for ref element, and exclude rest non-objects
        element_has_name = getattr(tree_element, 'name', None) is not None
        try:
            if not element_has_name: tree_element.name = 'plask-GUI--object-to-plot'
            manager = plask.Manager()
            try:
                manager.load(self.document.get_content(sections='geometry'))
                to_plot = manager.geometry[str(tree_element.name)]
                self.geometry_view.update_plot(to_plot)
            except Exception as e:
                QtGui.QMessageBox.critical(self.document.window, 'Error while interpreting XPL content.',
                                       "Geometry can not be plotted due to the error in XPL content:\n{}".format(str(e)))
        finally:
            if not element_has_name: tree_element.name = None

        #plot_geometry(current_index.internalPointer())

    def _construct_toolbar(self):
        toolbar = QtGui.QToolBar()
        toolbar.setStyleSheet("QToolBar { border: 0px }")

        self.add_menu = QtGui.QMenu()

        addButton = QtGui.QToolButton()
        addButton.setText('Add')
        addButton.setIcon(QtGui.QIcon.fromTheme('list-add'))
        addButton.setToolTip('Add new geometry object to the tree')
        addButton.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Plus)
        addButton.setMenu(self.add_menu)
        addButton.setPopupMode(QtGui.QToolButton.InstantPopup)
        toolbar.addWidget(addButton)

        self.remove_action = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove'), '&Remove', toolbar)
        self.remove_action.setStatusTip('Remove selected node from the tree')
        # self.remove_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Minus)
        self.remove_action.triggered.connect(self.remove_node)
        toolbar.addAction(self.remove_action)

        self.move_up_action = QtGui.QAction(QtGui.QIcon.fromTheme('go-up'), 'Move &up', toolbar)
        self.move_up_action.setStatusTip('Change order of entries: move current entry up')
        self.move_up_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Up)
        self.move_up_action.triggered.connect(self.move_current_up)
        toolbar.addAction(self.move_up_action)

        self.move_down_action = QtGui.QAction(QtGui.QIcon.fromTheme('go-down'), 'Move &down', toolbar)
        self.move_down_action.setStatusTip('Change order of entries: move current entry down')
        self.move_down_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Down)
        self.move_down_action.triggered.connect(self.move_current_down)
        toolbar.addAction(self.move_down_action)

        self.plot_action = QtGui.QAction(QtGui.QIcon.fromTheme('plask'), '&Plot', toolbar)
        self.plot_action.setStatusTip('Plot selected geometry object')
        self.plot_action.triggered.connect(self.plot)
        toolbar.addAction(self.plot_action)

        return toolbar

    def _construct_tree(self, model):
        self.tree = QtGui.QTreeView()
        self.tree.setModel(model)
        self.properties_delegate = HTMLDelegate(self.tree)
        self.tree.setItemDelegateForColumn(1, self.properties_delegate)
        self.tree.setColumnWidth(0, 200)
        return self.tree

    #def _construct_plot_dock(self):
    #    self.geometry_view = PlotWidget()
    #    self.document.window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.geometry_view.dock_window(self.document.window))

    def __init__(self, document, model=None):
        if model is None: model = GeometryModel()
        Controller.__init__(self, document, model)

        self._current_index = None
        self._last_index = None
        self._current_controller = None

        tree_with_buttons = QtGui.QGroupBox()
        vbox = QtGui.QVBoxLayout()
        tree_with_buttons.setLayout(vbox)

        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._construct_toolbar())
        vbox.addWidget(self._construct_tree(model))
        tree_selection_model = self.tree.selectionModel()   # workaround of segfault in pySide,
        # see http://stackoverflow.com/questions/19211430/pyside-segfault-when-using-qitemselectionmodel-with-qlistview
        tree_selection_model.selectionChanged.connect(self.grid_selected)
        self.update_actions()

        self.vertical_splitter = QtGui.QSplitter()
        self.vertical_splitter.setOrientation(QtCore.Qt.Vertical)

        self.vertical_splitter.addWidget(tree_with_buttons)

        self.parent_for_editor_widget = QtGui.QStackedWidget()
        self.vertical_splitter.addWidget(self.parent_for_editor_widget)

        self.geometry_view = PlotWidget()

        self.main_splitter = QtGui.QSplitter()
        self.main_splitter.addWidget(self.vertical_splitter)
        self.main_splitter.addWidget(self.geometry_view)

        #self._construct_plot_dock()

    def set_current_index(self, new_index):
        """
            Try to change current script.
            :param QtCore.QModelIndex new_index: index of new current script
            :return: False only when script should restore old selection
        """
        if self._current_index == new_index: return True
        if self._current_controller is not None:
            if not self._current_controller.on_edit_exit():
                return False
        self._current_index = new_index
        for i in reversed(range(self.parent_for_editor_widget.count())):
            self.parent_for_editor_widget.removeWidget(self.parent_for_editor_widget.widget(i))
        if self._current_index is None:
            self._current_controller = None
        else:
            self._current_controller = self._current_index.internalPointer().get_controller(self.document, self.model)
            self.parent_for_editor_widget.addWidget(self._current_controller.get_widget())
            self._current_controller.on_edit_enter()
        self.update_actions()
        return True

    def grid_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0] if indexes else None)):
            self.tree.selectionModel().select(old_selection, QtGui.QItemSelectionModel.ClearAndSelect)

    def on_edit_enter(self):
        self.tree.selectionModel().clear()   # model could completely changed
        #if self._last_index is not None:
        #    self.tree.selectRow(self._last_index)

    def on_edit_exit(self):
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.tree.selectionModel().clear()
        return True

    def get_widget(self):
        return self.main_splitter
