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
from ...qt.QtCore import Qt
from ...model.geometry import GeometryModel
from ...model.geometry.geometry import GNGeometryBase
from ...model.geometry.again_copy import GNAgain
from ...model.geometry.object import GNObject
from ...model.geometry.constructor import construct_by_name, construct_using_constructor
from ...model.geometry.types import geometry_types_geometries_core, gname

from .. import Controller
from ...utils.widgets import HTMLDelegate, VerticalScrollArea

from .plot_widget import PlotWidget


class GeometryController(Controller):
    # TODO use ControllerWithSubController (?)

    def _add_child(self, type_constructor, parent_index):
        parent = parent_index.internalPointer()
        pos = parent.new_child_pos()
        #self.model.beginInsertRows(parent_index, pos, pos)
        #new_node = type_constructor(None, None)
        #new_node.set_parent(parent)
        #construct_using_constructor(type_constructor, parent)
        #self.model.endInsertRows()
        self.model.insert_node(parent, type_constructor(None, None))
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
                a = QtGui.QAction(gname(type_name, True), result)
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
                self.add_menu.addAction('&Item').setMenu(add_child_menu)
        for n in geometry_types_geometries_core.keys():
            a = QtGui.QAction(gname(n, True), self.add_menu)
            a.triggered[()].connect(lambda n=n: self.append_geometry_node(n))
            self.add_menu.addAction(a)

    def update_actions(self):
        has_selected_object = not self.tree.selectionModel().selection().isEmpty()
        self.remove_action.setEnabled(has_selected_object)
        # self.plot_action.setEnabled(has_selected_object)
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
        self.update_actions()

    def remove_node(self):
        index = self.tree.selectionModel().currentIndex()
        model = self.tree.model()
        if model.removeRow(index.row(), index.parent()):
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

    def on_pick_object(self, event):
        self.tree.setCurrentIndex(
            self.model.index_for_node(self.plotted_tree_element.get_node_by_real_path(event.artist.plask_real_path))
        )
        #print ', '.join(str(p) for p in event.artist.plask_real_path)

    def plot_element(self, tree_element, set_limits=True):
        self.manager = plask.Manager()
        try:
            self.manager.load(self.document.get_content(sections='geometry'))
            #to_plot = self.manager.geometry[str(tree_element.ref if is_ref else tree_element.name)]
            self.plotted_object = self.model.fake_root.get_corresponding_object(tree_element, self.manager)
            self.geometry_view.update_plot(self.plotted_object, set_limits=set_limits, plane=self.checked_plane)
        except Exception as e:
            self.manager = None
            self.plotted_object = None
            self.status_bar.showMessage(str(e))
            palette = self.status_bar.palette()
            palette.setColor(QtGui.QPalette.Background, '#ff8888')
            self.status_bar.setPalette(palette)
            self.status_bar.setAutoFillBackground(True)
            from ... import _DEBUG
            if _DEBUG:
                import traceback
                traceback.print_exc()
            return False
        else:
            if tree_element.dim == 3:
                self.geometry_view.toolbar.enable_planes(tree_element.get_axes_conf())
            else:
                self.geometry_view.toolbar.disable_planes(tree_element.get_axes_conf())
            self.status_bar.showMessage('')
            palette = self.status_bar.palette()
            palette.setColor(QtGui.QPalette.Background, self.statusbar_color)
            self.status_bar.setPalette(palette)
            self.status_bar.setAutoFillBackground(False)
        return True

    def plot(self, tree_element=None):
        if tree_element is None:
            current_index = self.tree.selectionModel().currentIndex()
            if not current_index.isValid(): return
            tree_element = current_index.internalPointer()
        if self.plot_element(tree_element):
            self.plotted_tree_element = tree_element

    def on_model_change(self, *args, **kwargs):
        if self.plotted_tree_element is not None and self.plot_auto_refresh:
            self.plot_element(self.plotted_tree_element, set_limits=False)

    def _construct_toolbar(self):
        toolbar = QtGui.QToolBar()
        toolbar.setStyleSheet("QToolBar { border: 0px }")

        toolbar.addAction(self.model.create_undo_action(toolbar))
        toolbar.addAction(self.model.create_redo_action(toolbar))
        toolbar.addSeparator()

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
        self.remove_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Minus)
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

        toolbar.addSeparator()

        self.plot_auto_refresh = True

        return toolbar

    def _construct_tree(self, model):
        self.tree = QtGui.QTreeView()
        self.tree.setModel(model)
        self.properties_delegate = HTMLDelegate(self.tree)
        self.tree.setItemDelegateForColumn(1, self.properties_delegate)
        self.tree.setColumnWidth(0, 200)

        self.tree.setAutoScroll(True)

        self.tree.dragEnabled()
        self.tree.acceptDrops()
        self.tree.showDropIndicator()
        self.tree.setDragDropMode(QtGui.QAbstractItemView.InternalMove)

        return self.tree

    #def _construct_plot_dock(self):
    #    self.geometry_view = PlotWidget()
    #    self.document.window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.geometry_view.dock_window(self.document.window))

    def __init__(self, document, model=None):
        if model is None: model = GeometryModel()
        Controller.__init__(self, document, model)

        self.manager = None
        self.plotted_object = None

        self.plotted_tree_element = None
        self.model.changed.connect(self.on_model_change)

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
        tree_selection_model.selectionChanged.connect(self.object_selected)
        self.update_actions()

        self.checked_plane = '12'

        self.vertical_splitter = QtGui.QSplitter()
        self.vertical_splitter.setOrientation(QtCore.Qt.Vertical)

        self.vertical_splitter.addWidget(tree_with_buttons)

        self.parent_for_editor_widget = VerticalScrollArea()
        self.vertical_splitter.addWidget(self.parent_for_editor_widget)

        self.geometry_view = PlotWidget(self, picker=True)
        self.geometry_view.canvas.mpl_connect('pick_event', self.on_pick_object)

        self.status_bar = QtGui.QStatusBar()
        self.status_bar.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        palette = self.status_bar.palette()
        self.statusbar_color = palette.color(QtGui.QPalette.Background)
        self.status_bar.setPalette(palette)

        geometry_widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        geometry_widget.setLayout(layout)
        layout.addWidget(self.geometry_view)
        layout.addWidget(self.status_bar)

        self.main_splitter = QtGui.QSplitter()
        self.main_splitter.addWidget(self.vertical_splitter)
        self.main_splitter.addWidget(geometry_widget)

    def set_current_index(self, new_index):
        """
            Try to change current object.
            :param QtCore.QModelIndex new_index: index of new current object
            :return: False only when object should restore old selection
        """
        if self._current_index == new_index: return True
        if self._current_controller is not None:
            if not self._current_controller.on_edit_exit():
                return False
        self._current_index = new_index
        if self._current_index is None:
            self._current_controller = None
            self.parent_for_editor_widget.setWidget(QtGui.QWidget())
            self.vertical_splitter.moveSplitter(1, 0)
        else:
            self._current_controller = self._current_index.internalPointer().get_controller(self.document, self.model)
            widget = self._current_controller.get_widget()
            self.parent_for_editor_widget.setWidget(widget)
            widget.setFixedWidth(self.parent_for_editor_widget.size().width()-2)
            widget.update()
            h = widget.height()
            self.vertical_splitter.moveSplitter(self.vertical_splitter.height()-h-12, 1)
            self._current_controller.on_edit_enter()
        self.update_actions()

        #geometry_node = self.tree.selectionModel().currentIndex().internalPointer()
        try:
            plotted_root = self.plotted_tree_element.root
            current_root = self._current_index.internalPointer().root
        except AttributeError:
            pass
        else:
            if current_root != plotted_root:
                self.plot(current_root)
            if self.plotted_object is not None:
                to_select = self.model.fake_root.get_corresponding_object(self._current_index.internalPointer(),
                                                                          self.manager)
                bboxes = self.plotted_object.get_object_bboxes(to_select)
                self.geometry_view.clean_selectors()
                for b in bboxes: self.geometry_view.select_bbox(b)
                self.geometry_view.canvas.draw()
                # self.plot_action.setEnabled(isinstance(geometry_node, GNAgain) or isinstance(geometry_node, GNObject))

        return True

    def current_root(self):
        try:
            current_root = self._current_index.internalPointer().root
        except AttributeError:
            return None
        else:
            return current_root

    def zoom_to_current(self):
        if self.plotted_object is not None:
            to_select = self.model.fake_root.get_corresponding_object(self._current_index.internalPointer(),
                                                                      self.manager)
            bboxes = self.plotted_object.get_object_bboxes(to_select)
            if not bboxes: return
            box = bboxes[0]
            for b in bboxes[1:]:
                box += b
            self.geometry_view.zoom_bbox(box)

    def object_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0] if indexes else None)):
            self.tree.selectionModel().select(old_selection, QtGui.QItemSelectionModel.ClearAndSelect)

    def on_edit_enter(self):
        self.tree.selectionModel().clear()   # model could have been completely changed
        new_index = self.model.index(0, 0)
        self.tree.selectionModel().select(new_index,
                                          QtGui.QItemSelectionModel.Clear | QtGui.QItemSelectionModel.Select |
                                          QtGui.QItemSelectionModel.Rows)
        self.tree.setCurrentIndex(new_index)
        self.plot()

    def on_edit_exit(self):
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.tree.selectionModel().clear()
        return True

    def get_widget(self):
        return self.main_splitter
