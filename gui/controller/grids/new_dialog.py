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

from ...qt.QtCore import *
from ...qt.QtGui import *
from ...qt.QtWidgets import *
from ...qt import qt_exec
from ...model.grids import Grid
from ...model.grids.types import construct_grid, meshes_types, generators_types,\
    generator_methods, xml_name
from ...utils.widgets import ComboBox


class NewGridDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Mesh")

        kind = QGroupBox("Kind")
        self.kind_mesh = QRadioButton("&Mesh")
        self.kind_generator = QRadioButton("&Generator")
        self.kind_generator.toggled.connect(self._set_mode)
        self.kind_mesh.setChecked(True)
        hbox = QHBoxLayout()
        hbox.addWidget(self.kind_mesh)
        hbox.addWidget(self.kind_generator)
        #hbox.addStretch(1)    #??
        kind.setLayout(hbox)

        self.name_edit = QLineEdit()
        self.name_edit.setToolTip('Name of the mesh or generator for reference in configuration of the solvers.')

        self.type_edit = ComboBox()
        self.type_edit.setEditable(True)
        self.type_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.type_edit.editTextChanged.connect(self._type_changed)
        self.type_edit.setToolTip('Type of the mesh.')

        self.method_edit = ComboBox()
        self.method_edit.setEditable(True)
        self.method_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.method_edit.setToolTip('Generation method i.e. the type of the generator.')
        #self.method_edit_label = QLabel("Method:")

        self.form_layout = QFormLayout()
        self.form_layout.addRow("&Name:", self.name_edit)
        self.form_layout.addRow("&Type:", self.type_edit)
        self.form_layout.addRow("M&ethod:", self.method_edit)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |  QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addWidget(kind)
        main_layout.addLayout(self.form_layout)
        main_layout.addStretch()
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

        self._set_mode(False)

    def _set_mode(self, is_generator):
        self.form_layout.labelForField(self.method_edit).setVisible(is_generator)
        #self.method_edit.setEnabled(is_generator)
        #self.type_edit.clear()
        text = self.type_edit.currentText()
        if is_generator:
            self.type_edit.setModel(QStringListModel(sorted(generators_types())))
        else:
            self.type_edit.setModel(QStringListModel(sorted(meshes_types())))
        self.type_edit.setEditText(text)
        self.method_edit.setVisible(is_generator)
        self._type_changed(text)

    def _type_changed(self, new_type):
        if not self.method_edit.isVisible(): return
        text = self.method_edit.currentText()
        self.method_edit.setModel(QStringListModel(sorted(generator_methods(xml_name(new_type)))))
        self.method_edit.setEditText(text)

    def get_grid(self, grids_model):
        return construct_grid(
            grids_model,
            Grid.contruct_empty_xml_element(
                self.name_edit.text(),
                xml_name(self.type_edit.currentText()),
                xml_name(self.method_edit.currentText()) if self.kind_generator.isChecked() else None
            )
        )


def construct_grid_using_dialog(grids_model):
    dial = NewGridDialog()
    if qt_exec(dial) == QDialog.DialogCode.Accepted:
        return dial.get_grid(grids_model)
