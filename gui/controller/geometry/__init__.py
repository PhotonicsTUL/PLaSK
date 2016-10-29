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

import plask  # TODO: make preview optional

from ...qt import QtGui, QtCore, QtSlot
from ...qt.QtCore import Qt
from ...model.geometry import GeometryModel
from ...model.geometry.types import geometry_types_geometries_core, gname
from ...model.geometry.geometry import GNGeometryBase
from ...model.info import Info
from .. import Controller
from ...utils.widgets import HTMLDelegate, VerticalScrollArea, create_undo_actions

try:
    from .plot_widget import PlotWidget
except ImportError:
    PlotWidget = None


class GeometryTreeView(QtGui.QTreeView):

    def __init__(self, *args, **kwargs):
        super(GeometryTreeView, self).__init__(*args, **kwargs)
        self._current_index = None

    @QtSlot()
    def update_current_index(self):
        if self._current_index is not None:
            self.setCurrentIndex(self._current_index)
            self._current_index = None


class GeometryController(Controller):
    # TODO use ControllerWithSubController (?)

    def _add_child(self, type_constructor, parent_index):
        parent = parent_index.internalPointer()
        pos = parent.new_child_pos()
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
            for type_name, type_constructor in section.items():
                if type_name.endswith('2d') or type_name.endswith('3d'):
                    type_name = type_name[:-2]
                a = QtGui.QAction(gname(type_name, True), result)
                a.triggered[()].connect(lambda type_constructor=type_constructor, parent_index=geometry_node_index:
                                        self._add_child(type_constructor, parent_index))
                result.addAction(a)
        return result

    def _reparent(self, index, type_constructor):
        parent_index = index.parent()
        if not parent_index.isValid(): return
        new_index = self.model.reparent(index, type_constructor)
        self.tree.setExpanded(new_index, True)
        self.tree.selectionModel().select(new_index,
                                          QtGui.QItemSelectionModel.Clear | QtGui.QItemSelectionModel.Select |
                                          QtGui.QItemSelectionModel.Rows)
        self.tree.setCurrentIndex(new_index)
        self.update_actions()

    def _get_reparent_menu(self, index):
        parent_index = index.parent()
        if not parent_index.isValid(): return
        node = index.internalPointer()
        if node is None: return
        first = True
        result = QtGui.QMenu(self.more_menu)
        for section in node.add_parent_options(parent_index.internalPointer()):
            if not first:
                result.addSeparator()
            first = False
            for type_name, type_constructor in section.items():
                if type_name.endswith('2d') or type_name.endswith('3d'):
                    type_name = type_name[:-2]
                a = QtGui.QAction(gname(type_name, True), result)
                a.triggered[()].connect(lambda type_constructor=type_constructor, index=index:
                                        self._reparent(index, type_constructor))
                result.addAction(a)
        return result

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

        self._lims = None

        tree_with_buttons = QtGui.QGroupBox()
        vbox = QtGui.QVBoxLayout()
        tree_with_buttons.setLayout(vbox)

        tree_with_buttons.setTitle("Geometry Tree")
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
        self.vertical_splitter.setOrientation(Qt.Vertical)

        self.vertical_splitter.addWidget(tree_with_buttons)

        self.parent_for_editor_widget = VerticalScrollArea()
        self.vertical_splitter.addWidget(self.parent_for_editor_widget)

        self.main_splitter = QtGui.QSplitter()
        self.main_splitter.addWidget(self.vertical_splitter)

        search_action = QtGui.QAction(QtGui.QIcon.fromTheme('edit-find'), '&Search', self.main_splitter)
        search_action.setShortcut(QtGui.QKeySequence.Find)
        search_action.triggered.connect(lambda: self.search_box.setFocus())
        self.main_splitter.addAction(search_action)


        if PlotWidget is not None:
            self.geometry_view = PlotWidget(self, picker=True)
            self.geometry_view.canvas.mpl_connect('pick_event', self.on_pick_object)
            # self.status_bar = QtGui.QLabel()
            # self.status_bar.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark)")
            # self.geometry_view.layout().addWidget(self.status_bar)
            self.main_splitter.addWidget(self.geometry_view)

        self.document.window.config_changed.connect(self.reconfig)

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

    def fill_more_menu(self):
        self.more_menu.clear()
        current_index = self.tree.selectionModel().currentIndex()
        if current_index.isValid():
            reparent = self._get_reparent_menu(current_index)
            if reparent is not None and not reparent.isEmpty():
                self.more_menu.addMenu(reparent).setText("&Insert into")

    def update_actions(self):
        has_selected_object = not self.tree.selectionModel().selection().isEmpty()
        self.remove_action.setEnabled(has_selected_object)
        # self.plot_action.setEnabled(has_selected_object)
        self.fill_add_menu()

        u, d = self.model.can_move_node_up_down(self.tree.selectionModel().currentIndex())
        self.move_up_action.setEnabled(u)
        self.move_down_action.setEnabled(d)

        parent = self.tree.selectionModel().currentIndex().parent()
        self.duplicate_action.setEnabled(parent.isValid() and parent.internalPointer().accept_new_child())

        self.fill_more_menu()
        self.more_menu.parent().setEnabled(not self.more_menu.isEmpty())

    def append_geometry_node(self, type_name):
        self.tree.model().append_geometry(type_name)
        new_index = self.model.index(len(self.tree.model().roots)-1, 0)
        self.tree.selectionModel().select(new_index,
                                          QtGui.QItemSelectionModel.Clear | QtGui.QItemSelectionModel.Select |
                                          QtGui.QItemSelectionModel.Rows)
        self.tree.setCurrentIndex(new_index)
        self.update_actions()

    def on_tree_context_menu(self, pos):
        index = self.tree.indexAt(pos)
        if not index.isValid(): return

        menu = QtGui.QMenu(self.tree)
        add_child_menu = self._get_add_child_menu(index)
        if add_child_menu:
            menu.addMenu(add_child_menu).setText("&Add item")
        menu.addAction("&Remove", lambda: self.remove_node(index))
        u, d = self.model.can_move_node_up_down(index)
        if u: menu.addAction("Move &up", lambda: self.move_up(index))
        if d: menu.addAction("Move d&own", lambda: self.move_down(index))

        parent = index.parent()
        if parent.isValid():
            parent_node = parent.internalPointer()
            if parent_node.accept_new_child():
                menu.addAction("&Duplicate", lambda: self.duplicate(index))

        reparent_menu = self._get_reparent_menu(index)
        if reparent_menu and not reparent_menu.isEmpty():
            if not menu.isEmpty(): menu.addSeparator()
            menu.addMenu(reparent_menu).setText("&Insert into")

        menu.exec_(self.tree.mapToGlobal(pos))

    def remove_node(self, index):
        model = self.tree.model()
        if model.removeRow(index.row(), index.parent()):
            self.update_actions()

    def remove_current_node(self):
        self.remove_node(self.tree.selectionModel().currentIndex())

    def _swap_neighbour_nodes(self, parent_index, row1, row2):
        if self.model.is_read_only(): return
        if row2 < row1: row1, row2 = row2, row1
        children = self.model.children_list(parent_index)
        if row1 < 0 or row2 < len(children): return
        self.model.beginMoveRows(parent_index, row2, row2, parent_index, row1)
        children[row1], children[row2] = children[row2], children[row1]
        self.model.endMoveRows()
        self.fire_changed()

    def move_up(self, index):
        self.model.move_node_up(index)
        self.update_actions()

    def move_current_up(self):
        self.move_up(self.tree.selectionModel().currentIndex())

    def move_down(self, index):
        self.model.move_node_down(index)
        self.update_actions()

    def move_current_down(self):
        self.move_down(self.tree.selectionModel().currentIndex())

    def duplicate(self, index):
        self.model.duplicate(index)
        self.update_actions()

    def duplicate_current(self):
        return self.duplicate(self.tree.selectionModel().currentIndex())

    def on_pick_object(self, event):
        # This seems as an ugly hack, but in reality this is the only way to make sure
        # that `setCurrentIndex` is called only once if there are multiple artists in
        # the clicked spot.
        self.tree._current_index = self.model.index_for_node(
            self.plotted_tree_element.get_node_by_real_path(event.artist.plask_real_path))
        QtCore.QMetaObject.invokeMethod(self.tree, 'update_current_index', Qt.QueuedConnection)

    def plot_element(self, tree_element, set_limits):
        manager = plask.Manager(draft=True)
        try:
            manager.load(self.document.get_content(sections=('defines', 'geometry')))
            try:
                plotted_object = self.model.fake_root.get_corresponding_object(tree_element, manager)
            except ValueError:
                return False
            if tree_element != self.plotted_tree_element:
                self.geometry_view.toolbar._views.clear()
            self.geometry_view.update_plot(plotted_object, set_limits=set_limits, plane=self.checked_plane)
        except Exception as e:
            self.model.info_message("Could not update geometry view: {}".format(str(e)), Info.WARNING)
            # self.status_bar.setText(str(e))
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark); background-color: #ff8888;")
            from ... import _DEBUG
            if _DEBUG:
                import traceback
                traceback.print_exc()
                sys.stderr.flush()
            return False
        else:
            self.manager = manager
            self.plotted_tree_element = tree_element
            self.plotted_object = plotted_object
            if tree_element.dim == 3:
                self.geometry_view.toolbar.enable_planes(tree_element.get_axes_conf())
            else:
                self.geometry_view.toolbar.disable_planes(tree_element.get_axes_conf())
            self.model.info_message()
            # self.status_bar.setText('')
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark); background-color: palette(background);")
            self.show_selection()
            return True

    def plot(self, tree_element=None):
        if tree_element is None:
            current_index = self.tree.selectionModel().currentIndex()
            if not current_index.isValid(): return
            tree_element = current_index.internalPointer()
        self.plot_element(tree_element, set_limits=True)
        self.geometry_view.toolbar.clear_history()
        self.geometry_view.toolbar.push_current()

    def plot_refresh(self):
        self.plot_element(self.plotted_tree_element, set_limits=False)

    def reconfig(self):
        if self.plotted_tree_element is not None and self.get_widget().isVisible():
            self.plot_refresh()

    def on_model_change(self, *args, **kwargs):
        if self.plotted_tree_element is not None:
            if self.plot_auto_refresh:
                self.plot_refresh()
            else:
                self.model.info_message("Geometry changed: click here to refresh the plot", Info.INFO,
                                        action=self.plot_refresh)
                # self.status_bar.setText("Geometry changed: refresh the plot")
                # self.status_bar.setStyleSheet("border: 1px solid palette(dark); background-color: #ffff88;")
        else:
            self.model.info_message()
            # self.status_bar.setText('')
            # self.status_bar.setStyleSheet("border: 1px solid palette(dark); background-color: palette(background);")

    #def _construct_plot_dock(self):
    #    self.geometry_view = PlotWidget()
    #    self.document.window.addDockWidget(Qt.RightDockWidgetArea, self.geometry_view.dock_window(self.document.window))

    def _construct_toolbar(self):
        toolbar = QtGui.QToolBar()
        toolbar.setStyleSheet("QToolBar { border: 0px }")

        create_undo_actions(toolbar, self.model, toolbar)
        toolbar.addSeparator()

        self.add_menu = QtGui.QMenu()

        add_button = QtGui.QToolButton()
        add_button.setText('Add')
        add_button.setIcon(QtGui.QIcon.fromTheme('list-add'))
        add_button.setToolTip('Add new geometry object to the tree')
        add_button.setShortcut(Qt.CTRL + Qt.Key_Plus)
        add_button.setMenu(self.add_menu)
        add_button.setPopupMode(QtGui.QToolButton.InstantPopup)
        toolbar.addWidget(add_button)

        self.remove_action = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove'), '&Remove', toolbar)
        self.remove_action.setStatusTip('Remove selected node from the tree')
        self.remove_action.setShortcut(Qt.SHIFT + Qt.Key_Delete)
        self.remove_action.triggered.connect(self.remove_current_node)
        toolbar.addAction(self.remove_action)

        self.move_up_action = QtGui.QAction(QtGui.QIcon.fromTheme('go-up'), 'Move &up', toolbar)
        self.move_up_action.setStatusTip('Change order of entries: move current entry up')
        self.move_up_action.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_Up)
        self.move_up_action.triggered.connect(self.move_current_up)
        toolbar.addAction(self.move_up_action)

        self.move_down_action = QtGui.QAction(QtGui.QIcon.fromTheme('go-down'), 'Move d&own', toolbar)
        self.move_down_action.setStatusTip('Change order of entries: move current entry down')
        self.move_down_action.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_Down)
        self.move_down_action.triggered.connect(self.move_current_down)
        toolbar.addAction(self.move_down_action)

        self.duplicate_action = QtGui.QAction(QtGui.QIcon.fromTheme('edit-copy'), '&Duplicate', toolbar)
        self.duplicate_action.setStatusTip('Duplicate current entry and insert it '
                                           'into default position of the same container')
        self.duplicate_action.setShortcut(Qt.CTRL + Qt.Key_D)
        self.duplicate_action.triggered.connect(self.duplicate_current)
        toolbar.addAction(self.duplicate_action)

        toolbar.addSeparator()

        more_button = QtGui.QToolButton(toolbar)
        more_button.setIcon(QtGui.QIcon.fromTheme('menu-other'))
        self.more_menu = QtGui.QMenu(more_button)
        more_button.setMenu(self.more_menu)
        more_button.setPopupMode(QtGui.QToolButton.InstantPopup)
        toolbar.addWidget(more_button)

        toolbar.addSeparator()
        spacer = QtGui.QWidget()
        spacer.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        toolbar.addWidget(spacer)
        self.search_combo = QtGui.QComboBox()
        self.search_combo.setEditable(True)
        self.search_combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        search_box = self.search_combo.lineEdit()
        # search_box.setAlignment(Qt.AlignRight)
        search_box.setPlaceholderText("Name search")
        search_box.returnPressed.connect(self.search)
        self.search_combo.currentIndexChanged.connect(lambda i: self.search())
        self.search_combo.setMinimumWidth(search_box.sizeHint().width())
        toolbar.addWidget(self.search_combo)
        find_action = QtGui.QAction(QtGui.QIcon.fromTheme('edit-find'), '&Find', toolbar)
        find_action.triggered.connect(self.search)
        toolbar.addAction(find_action)

        self.model.dataChanged.connect(lambda i,j: self._fill_search_combo())

        self.plot_auto_refresh = True
        return toolbar

    def _construct_tree(self, model):
        self.tree = GeometryTreeView()
        self.tree.setModel(model)
        self.properties_delegate = HTMLDelegate(self.tree)
        self.tree.setItemDelegateForColumn(1, self.properties_delegate)
        self.tree.setColumnWidth(0, 200)

        self.tree.setAutoScroll(True)

        self.tree.dragEnabled()
        self.tree.acceptDrops()
        self.tree.showDropIndicator()
        self.tree.setDragDropMode(QtGui.QAbstractItemView.DragDrop)
        self.tree.setDefaultDropAction(Qt.MoveAction)

        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.on_tree_context_menu)

        return self.tree

    def _fill_search_combo(self):
        self.search_combo.clear()
        self.search_combo.addItems([''] + list(self.model.names()))

    def show_selection(self):
        if self._current_index is None: return
        node = self._current_index.internalPointer()
        if isinstance(node, GNGeometryBase):
            self.geometry_view.clean_selectors()  # TODO: show geometry edges
            self.geometry_view.canvas.draw()
        else:
            selected = self.model.fake_root.get_corresponding_object(node, self.manager)
            if selected is not None:
                self.geometry_view.select_object(self.plotted_object, selected)
            else:
                self.geometry_view.clean_selectors()
                self.geometry_view.canvas.draw()

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
                self.show_selection()
                # self.plot_action.setEnabled(isinstance(geometry_node, GNAgain) or isinstance(geometry_node, GNObject))

        return True

    def search(self):
        text = self.search_combo.currentText()
        if not text:
            return
        found = self.model.index_for_node(self.model.find_by_name(text))
        if found and found.isValid():
            self.tree.setCurrentIndex(found)
            self.search_combo.setEditText("")
            self.tree.setFocus()
        else:
            red = QtGui.QPalette()
            red.setColor(QtGui.QPalette.Text, QtGui.QColor("#a00"))
            pal = self.search_combo.palette()
            self.search_combo.setPalette(red)
            QtCore.QTimer.singleShot(500, lambda: self.search_combo.setPalette(pal))

    def current_root(self):
        try:
            current_root = self._current_index.internalPointer().root
        except AttributeError:
            return None
        else:
            return current_root

    def zoom_to_current(self):
        if self.plotted_object is not None:
            obj = self.model.fake_root.get_corresponding_object(self._current_index.internalPointer(),
                                                                self.manager)
            bboxes = self.plotted_object.get_object_bboxes(obj)
            if not bboxes: return
            box = bboxes[0]
            for b in bboxes[1:]:
                box += b
            self.geometry_view.zoom_bbox(box)

    def zoom_to_root(self):
        if self.plotted_object is not None:
            box = self.plotted_object.bbox
            self.geometry_view.zoom_bbox(box)

    def object_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0] if indexes else None)):
            self.tree.selectionModel().select(old_selection, QtGui.QItemSelectionModel.ClearAndSelect)

    def on_edit_enter(self):
        self.tree.selectionModel().clear()   # model could have been completely changed
        if self.model.dirty:
            self._last_index = None
            self.model.dirty = False
        try:
            if not self._last_index:
                raise IndexError(self._last_index)
            new_index = self._last_index
            self.tree.selectionModel().select(new_index,
                                              QtGui.QItemSelectionModel.Clear | QtGui.QItemSelectionModel.Select |
                                              QtGui.QItemSelectionModel.Rows)
            self.tree.setCurrentIndex(new_index)
            self.plot(self.plotted_tree_element)
            if self._lims is not None:
                self.geometry_view.axes.set_xlim(self._lims[0])
                self.geometry_view.axes.set_ylim(self._lims[1])
            self.show_selection()
        except IndexError:
            new_index = self.model.index(0, 0)
            self.tree.selectionModel().select(new_index,
                                              QtGui.QItemSelectionModel.Clear | QtGui.QItemSelectionModel.Select |
                                              QtGui.QItemSelectionModel.Rows)
            self.tree.setCurrentIndex(new_index)
            self.plot()
        self._fill_search_combo()
        self.update_actions()
        self.tree.setFocus()

    def on_edit_exit(self):
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.tree.selectionModel().clear()
            self._lims = self.geometry_view.axes.get_xlim(), self.geometry_view.axes.get_ylim()
        return True

    def get_widget(self):
        return self.main_splitter

    def select_info(self, info):
        try: action = info.action
        except AttributeError: pass
        else: return action()
        try:
            new_nodes = info.nodes
        except AttributeError:
            return
        if not new_nodes: return # empty??

        if len(new_nodes) == 1:
            self.tree.setCurrentIndex(self.model.index_for_node(new_nodes[0]))
        else:
            try:
                current_node = self.model.node_for_index(self.tree.currentIndex())
            except:
                self.tree.setCurrentIndex(self.model.index_for_node(new_nodes[0]))
            else:
                after = False
                found = False
                for n in new_nodes:
                    if after:
                        self.tree.setCurrentIndex(self.model.index_for_node(n))
                        found = True
                        break
                    if n == current_node: after = True
                if not found:
                    self.tree.setCurrentIndex(self.model.index_for_node(new_nodes[0]))
        try:    # try to set focus on proper widget
            self._current_controller.select_info(info)
        except AttributeError:
            pass

