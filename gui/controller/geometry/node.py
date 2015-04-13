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

from ...utils.str import empty_to_none, none_to_empty


class GNodeController(Controller):

    class SetTextPropertyCommand(QtGui.QUndoCommand):

        def __init__(self, model, node, property_name, new_value, display_property_name = None, unit = '', QUndoCommand_parent = None):
            if display_property_name is None: display_property_name = property_name
            super(GNodeController.SetTextPropertyCommand, self).__init__(u'change {} to "{}"{}'.format(display_property_name, new_value, none_to_empty(unit)), QUndoCommand_parent)
            self.model = model
            self.node = node
            self.property_name = property_name
            self.new_value = empty_to_none(new_value)
            self.old_value = getattr(node, property_name)

        def set_property_value(self, value):
            setattr(self.node, self.property_name, value)
            index = self.model.index_for_node(self.node, column=1)
            self.model.dataChanged.emit(index, index)
            self.model.fire_changed()

        def redo(self):
            self.set_property_value(self.new_value)

        def undo(self):
            self.set_property_value(self.old_value)

    def _set_node_property_undoable(self, property_name, new_value, display_property_name = None, unit = '', node=None):
        cmd = GNodeController.SetTextPropertyCommand(self.model, self.node if node is None else node,
                                                     property_name, new_value, display_property_name, unit)
        if cmd.new_value != cmd.old_value: self.model.undo_stack.push(cmd)


    def _get_current_form(self):
        if not hasattr(self, '_current_form'): self.construct_group()
        return self._current_form

    def construct_line_edit(self, row_name=None, use_defines_completer=True, unit=None, node_property_name = None, display_property_name = None):
        res = QtGui.QLineEdit()
        if use_defines_completer: res.setCompleter(self.defines_completer)
        if row_name:
            if unit is not None:
                box, _ = self._construct_hbox(row_name)
                box.addWidget(res)
                box.addWidget(QtGui.QLabel(unit))
            else:
                self._get_current_form().addRow(row_name, res)
        if node_property_name is None:
            res.editingFinished.connect(self.after_field_change)
        else:
            res.editingFinished.connect(lambda :
                self._set_node_property_undoable(node_property_name, res.text(), display_property_name, unit)
            )
        return res

    def construct_combo_box(self, row_name=None, items=[], editable=True, node_property_name = None, display_property_name = None, node = None):
        res = QtGui.QComboBox()
        res.setEditable(editable)
        res.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        res.addItems(items)
        if row_name: self._get_current_form().addRow(row_name, res)
        if node_property_name is None:
            res.editTextChanged.connect(self.after_field_change)
        else:
            res.editTextChanged.connect(lambda :
                self._set_node_property_undoable(node_property_name, res.currentText(), display_property_name, node=node)
            )
        return res

    def construct_material_combo_box(self, row_name = None, items = None, node_property_name = None, display_property_name = None, node = None):
        res = MaterialsComboBox()
        res.setEditable(True)
        res.append_list(items)
        res.append_materials_from_model(self.document.materials.model)
        res.append_materials_from_db()
        res.setMinimumWidth(2)
        if row_name: self._get_current_form().addRow(row_name, res)
        if node_property_name is None:
            res.editTextChanged.connect(self.after_field_change)
        else:
            res.editTextChanged.connect(lambda :
                self._set_node_property_undoable(node_property_name, res.currentText(), display_property_name, node=node)
            )
        return res

    def construct_names_before_self_combo_box(self, row_name = None, node_property_name = None, display_property_name = None):
        return self.construct_combo_box(row_name,
                                        items=[''] +
                                        sorted(self.model.names_before(self.node), key=lambda s: s.lower()),
                                        node_property_name=node_property_name,
                                        display_property_name=display_property_name)

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

    def construct_align_controllers(self, dim=None, add_to_current=True, *aligners_dir):
        """:return: list of controllers pairs, first is combo box to select aligner type,
                    second is line edit for its value"""
        if len(aligners_dir) == 0:
            return self.construct_align_controllers(dim, add_to_current, *self.node.aligners_dir())
        if dim is None: dim = self.node.children_dim
        positions = []
        layout = QtGui.QGridLayout(None)
        layout.setContentsMargins(0, 0, 0, 0)
        for r,c in enumerate(aligners_dir):
            axis = QtGui.QLabel(('Longitudinal:', 'Transverse:', 'Vertical:')[c+3-dim])
            layout.addWidget(axis, r, 0)
            position = QtGui.QComboBox()
            position.addItems(GNAligner.display_names(dim, c))
            layout.addWidget(position, r, 1)
            layout.addWidget(QtGui.QLabel('at'), r, 2)
            position.currentIndexChanged.connect(self.after_field_change)
            pos_value = self.construct_line_edit()
            layout.addWidget(pos_value, r, 3)
            positions.append((position, pos_value))
            layout.addWidget(QtGui.QLabel(u'µm'), r, 4)
        if add_to_current: self._get_current_form().addRow(layout)
        return positions

    def _construct_hbox(self, row_name=None):
        hbox = QtGui.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        group = QtGui.QWidget(self.form)
        group.setContentsMargins(0, 0, 0, 0)
        group.setLayout(hbox)
        if row_name:
            self._get_current_form().addRow(row_name, group)
        return hbox, group

    def construct_point_controllers(self, row_name=None, dim=None):
        if dim is None: dim = self.node.dim
        hbox, group = self._construct_hbox(row_name)
        res = tuple(self.construct_line_edit() for _ in range(0, dim))
        for i in range(0, dim):
            hbox.addWidget(res[i])
            hbox.addWidget(QtGui.QLabel(u'µm' + ('' if i == dim-1 else u'  × ')))
        if row_name:
            return res
        else:
            return res, group

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
        self.fill_form()
        try:
            del self._current_form
        except:
            pass

    @property
    def node_index(self):
        self.model.index_for_node(self.node)

    def fill_form(self):
        pass

    def after_field_change(self):
        self.save_data_in_model()
        index = self.node_index
        self.model.dataChanged.emit(index, index)
        self.model.fire_changed()

    def save_data_in_model(self):
        pass

    def _fill_form_using_data_from_model(self, *args, **kwargs):
        self.on_edit_enter()

    def on_edit_enter(self):
        self.model.changed.connect(self._fill_form_using_data_from_model)
        pass

    def on_edit_exit(self):
        self.model.changed.disconnect(self._fill_form_using_data_from_model)
        return super(GNodeController, self).on_edit_exit()

    def get_widget(self):
        return self.form


class GNChildController(GNodeController):

    def __init__(self, document, model, node, child_node):
        self.child_node = child_node
        super(GNChildController, self).__init__(document, model, node)
        self.vbox.setContentsMargins(0, 0, 0, 0)
