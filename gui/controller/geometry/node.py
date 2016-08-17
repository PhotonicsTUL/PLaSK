# coding=utf-8
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
from ...qt import QtGui

from .. import Controller
from ..materials import MaterialsComboBox
from ..defines import get_defines_completer
from ...model.geometry.reader import GNAligner

from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty
from ...utils.widgets import ComboBox, MultiLineEdit
from ...utils import getattr_by_path


try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str


def controller_to_aligners(position_controllers):
    """
    Convert data from widgets which allow to set aligner parameters to list of GNAligner.
    :param position_controllers: widgets with aligner parameters
    :return list[GNAligner]: resulted list of GNAligner
    """
    aligners_list = []
    for i, pos in enumerate(position_controllers):
        aligners_list.append(GNAligner(pos[0].currentIndex(), empty_to_none(pos[1].text())))
    return aligners_list


def aligners_to_controllers(aligners_list, position_controllers):
    """
    Get data from list of GNAligners and fill with them widgets which allow to set aligner parameters.
    :param list[GNAligner] aligners_list: source data
    :param position_controllers: widgets to fill
    """
    if aligners_list is None: return
    for i, pos in enumerate(position_controllers):
        aligner = aligners_list[i]
        with BlockQtSignals(pos[0], pos[1]):
            if aligner.position is not None:
                pos[0].setCurrentIndex(aligner.position)
            pos[1].setText(none_to_empty(aligner.value))


class GNodeController(Controller):

    class ChangeNodeCommand(QtGui.QUndoCommand):

        def __init__(self, model, node, setter, new_value, old_value, action_name, parent=None):
            super(GNodeController.ChangeNodeCommand, self).__init__(action_name, parent)
            self.model = model
            self.node = node
            self.setter = setter
            self.new_value = new_value
            self.old_value = old_value

        def set_property_value(self, value):
            self.setter(self.node, value)
            index = self.model.index_for_node(self.node, column=1)
            self.model.dataChanged.emit(index, index)
            self.model.fire_changed()

        def redo(self):
            self.set_property_value(self.new_value)

        def undo(self):
            self.set_property_value(self.old_value)

    def _set_node_property_undoable(self, property_name, new_value, display_property_name=None,
                                    unit='', node=None, action_name=None):
        if node is None: node = self.node
        if isinstance(new_value, basestring): new_value = empty_to_none(new_value)
        old_value = getattr(node, property_name)
        if action_name is None:
            if display_property_name is None: display_property_name = property_name
            action_name = u'change {} to "{}"{}'.format(display_property_name, none_to_empty(new_value), none_to_empty(unit))
        self._set_node_by_setter_undoable(lambda n, v: setattr(n, property_name, v), new_value, old_value,
                                          action_name=action_name, node=node)

    def _set_node_by_setter_undoable(self, setter, new_value, old_value, action_name, node=None):
        if new_value != old_value:
            self.model.undo_stack.push(GNodeController.ChangeNodeCommand(
                self.model, self.node if node is None else node,
                setter, new_value, old_value, action_name
            ))

    def _get_current_form(self):
        if not hasattr(self, '_current_form'): self.construct_group()
        return self._current_form

    def construct_line_edit(self, row_name=None, use_defines_completer=True, unit=None,
                            node_property_name=None, display_property_name=None, change_cb=None):
        res = QtGui.QLineEdit()
        if use_defines_completer: res.setCompleter(self.defines_completer)
        if row_name:
            if unit is not None:
                box, _ = self._construct_hbox(row_name)
                box.addWidget(res)
                box.addWidget(QtGui.QLabel(unit))
            else:
                self._get_current_form().addRow(row_name, res)
        if change_cb is not None:
            res.editingFinished.connect(change_cb)
        elif node_property_name is not None:
            res.editingFinished.connect(lambda :
                self._set_node_property_undoable(node_property_name, res.text(), display_property_name, unit))
        return res

    def construct_multi_line_edit(self, row_name=None, node_property_name=None, display_property_name=None,
                                  change_cb=None, sep='\n'):
        res = MultiLineEdit(change_cb=change_cb)
        # res = TextEditWithCB(key_cb=key_cb)
        # res.setTabChangesFocus(True)
        # res.setFixedHeight(int(3.5 * QtGui.QFontMetrics(res.font()).height()))
        if row_name: self._get_current_form().addRow(row_name, res)
        if change_cb is not None:
            res.focus_out_cb = change_cb
        elif node_property_name is not None:
            res.focus_out_cb = lambda: self._set_node_property_undoable(node_property_name, sep.join(res.get_values()),
                                                                        display_property_name)
        return res

    def construct_combo_box(self, row_name=None, items=(), editable=True, node_property_name=None,
                            display_property_name=None, node=None, change_cb=None):
        res = ComboBox()
        res.setEditable(editable)
        res.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        res.addItems(items)
        if row_name: self._get_current_form().addRow(row_name, res)
        if change_cb is not None:
            res.editingFinished.connect(change_cb)
        elif node_property_name is not None:
            res.editingFinished.connect(lambda:
                                        self._set_node_property_undoable(node_property_name, res.currentText(),
                                                                         display_property_name, node=node))
        return res

    def construct_material_combo_box(self, row_name=None, items=None, node_property_name=None,
                                     display_property_name=None, node=None, change_cb=None):
        if change_cb is None and node_property_name is not None:
            def change_cb():
                return self._set_node_property_undoable(
                    node_property_name, res.currentText(), display_property_name, node=node)
        res = MaterialsComboBox(materials_model=self.document.materials.model, popup_select_cb=lambda m: change_cb())
        res.setEditable(True)
        res.append_list(items)
        res.append_materials_from_model(self.document.materials.model)
        res.append_materials_from_db()
        res.setMinimumWidth(2)
        if row_name: self._get_current_form().addRow(row_name, res)
        if change_cb is not None:
            res.editingFinished.connect(change_cb)
        return res

    def construct_names_before_self_combo_box(self, row_name=None, node_property_name=None,
                                              display_property_name=None, change_cb=None):
        res = self.construct_combo_box(items=[''] +
                                       sorted(self.model.names_before(self.node), key=lambda s: s.lower()),
                                       node_property_name=node_property_name,
                                       display_property_name=display_property_name,
                                       change_cb=change_cb)

        def goto():
            found = self.model.find_by_name(res.currentText())
            if found is not None:
                self.document.geometry.gui.tree.setCurrentIndex(self.model.index_for_node(found))

        button = QtGui.QToolButton()
        button.setText("->")
        button.setToolTip("Go to selected object")
        button.pressed.connect(goto)

        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(res)
        layout.addWidget(button)
        widget = QtGui.QWidget()
        widget.setLayout(layout)

        if row_name:
            self._get_current_form().addRow(row_name, widget)
            return res
        else:
            return widget

    def construct_group(self, title=None, position=None):
        external = QtGui.QGroupBox(self.form)
        form_layout = QtGui.QFormLayout(external)
        if title is not None:
            external.setTitle(title)
        #     m = form_layout.getContentsMargins()
        #     form_layout.setContentsMargins(0, m[1], 0, 0)
        # else:
        #     form_layout.setContentsMargins(0, 0, 0, 0)
        if position is None:
            self.vbox.addWidget(external)
        else:
            self.vbox.insertWidget(position, external)
        external.setLayout(form_layout)
        self._current_form = form_layout
        return form_layout

    def construct_align_controllers(self, dim=None, add_to_current=True, aligners_dir=None, change_cb=None):
        """:return List[(QtGui.QComboBox, QtGui.QLineEdit)]: list of controllers pairs, first is combo box to select aligner type,
                    second is line edit for its value"""
        if aligners_dir is None:
            aligners_dir = self.node.aligners_dir()
            if aligners_dir is None: return
        if dim is None: dim = self.node.children_dim
        positions = []
        layout = QtGui.QGridLayout(None)
        layout.setContentsMargins(0, 0, 0, 0)
        for r, c in enumerate(aligners_dir):
            axis = QtGui.QLabel(('Longitudinal:', 'Transverse:', 'Vertical:')[c+3-dim])
            layout.addWidget(axis, r, 0)
            position = QtGui.QComboBox()
            position.addItems(GNAligner.display_names(dim, c))
            layout.addWidget(position, r, 1)
            layout.addWidget(QtGui.QLabel('at'), r, 2)
            pos_value = self.construct_line_edit()
            layout.addWidget(pos_value, r, 3)
            positions.append((position, pos_value))
            layout.addWidget(QtGui.QLabel(u'µm'), r, 4)
        if change_cb is not None:
            cb = lambda: change_cb(controller_to_aligners(positions))
            for p in positions:
                p[0].currentIndexChanged.connect(cb)
                p[1].editingFinished.connect(cb)
        if add_to_current: self._get_current_form().addRow(layout)
        return positions

    def _construct_hbox(self, row_name=None):
        hbox = QtGui.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        group = QtGui.QWidget(self.form)
        group.setContentsMargins(0, 0, 0, 0)
        group.setLayout(hbox)
        if row_name: self._get_current_form().addRow(row_name, group)
        return hbox, group

    def construct_point_controllers(self, row_name=None, dim=None, change_cb=None):
        """
        :param row_name:
        :param int dim: number of dims. (self.node.dim by default)
        :param change_cb: callable to which point as tuple will be given as argument
        :return:
        """
        if dim is None: dim = self.node.dim
        hbox, group = self._construct_hbox(row_name)
        res = tuple(self.construct_line_edit() for _ in range(0, dim))
        for i in range(0, dim):
            hbox.addWidget(res[i])
            hbox.addWidget(QtGui.QLabel(u'µm' + ('' if i == dim-1 else u'  × ')))
            if change_cb is not None:
                res[i].editingFinished.connect(lambda : change_cb(tuple(empty_to_none(p.text()) for p in res)))
        return res if row_name else (res, group)

    def __init__(self, document, model, node):
        """
        Construct node controller,
        :param XPLDocument document:
        :param GeometryModel model:
        :param model.geometry.node.GNode node:
        """
        super(GNodeController, self).__init__(document=document, model=model)
        self.node = node

        self.defines_completer = get_defines_completer(document.defines.model, None)

        self.form = QtGui.QWidget()

        self.vbox = QtGui.QVBoxLayout()
        self.vbox.setSpacing(0)
        self.form.setLayout(self.vbox)
        self.construct_form()
        try:
            del self._current_form
        except:
            pass

    @property
    def node_index(self):
        self.model.index_for_node(self.node)

    def construct_form(self):
        pass

    def fill_form(self):
        #self.on_edit_enter()
        pass

    def _fill_form_cb(self, *args, **kwargs):
        self.fill_form()

    def on_edit_enter(self):
        super(GNodeController, self).on_edit_enter()
        self.model.changed.connect(self._fill_form_cb)
        self.fill_form()

    def on_edit_exit(self):
        self.model.changed.disconnect(self._fill_form_cb)
        return super(GNodeController, self).on_edit_exit()

    def get_widget(self):
        return self.form

    def select_info(self, info):
        prop = getattr(info, "property", None)
        widget = getattr_by_path(self, prop, None)
        if widget is not None: widget.setFocus()


class GNChildController(GNodeController):

    def __init__(self, document, model, node, child_node):
        self.child_node = child_node
        super(GNChildController, self).__init__(document, model, node)
        self.vbox.setContentsMargins(0, 0, 0, 0)
