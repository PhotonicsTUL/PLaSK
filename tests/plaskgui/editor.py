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

from gui.qt.QtWidgets import *
from gui.qt.QtGui import *
from gui.qt import qt_exec
from gui.model.solvers import Tag
from gui.utils.widgets import MultiLineEdit


def open_editor(data, document):
    dialog = QDialog()
    layout = QVBoxLayout()
    attr = QLineEdit()
    layout.addWidget(attr)
    items = MultiLineEdit(movable=True, document=document)
    layout.addWidget(items)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setLayout(layout)

    attr.setText(data.attrs.get('attr'))
    items.set_values(t.name for t in data.tags)

    if qt_exec(dialog) == QDialog.Accepted:
        a = attr.text()
        return Tag(data.name, [Tag(i) for i in items.get_values()], {'attr': a} if a else {})
