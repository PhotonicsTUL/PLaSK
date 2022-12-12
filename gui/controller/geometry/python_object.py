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

# Copyright (C) 2021 Photonics Group, Lodz University of Technology
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

import weakref

from ...qt.QtCore import *
from ...qt.QtGui import *
from ...qt.QtWidgets import *
from ...qt import QtSignal
from .object import GNObjectController
from ...utils.texteditor.python import PythonEditor, PythonEditorWidget
from ...utils.qsignals import BlockQtSignals
from ...utils.config import CONFIG
from ...lib.highlighter.plask import SYNTAX, get_syntax
from ...import APPLICATION

SYNTAX['formats']['geometry_object'] = '{syntax_solver}'


WINDOW_TITLE = "Python Geometry Object"


class _PythonEditorWidow(QMainWindow):

    def __init__(self, controller, parent=None):
        self.controller = controller
        self.config_changed = controller.document.window.config_changed
        super().__init__(parent)

    def commit(self):
        self.controller.editor.editingFinished.emit()
        editor = self.centralWidget().editor
        editor.document().setModified(False)
        editor.modificationChanged.emit(False)

    def event(self, event):
        if event.type() == QEvent.Type.WindowDeactivate:
            self.commit()
        return super().event(event)

    def closeEvent(self, event):
        super().closeEvent(event)
        self.controller.button.setChecked(False)


class GNPythonController(GNObjectController):

    have_mesh_settings = False

    def construct_form(self):
        self.construct_group('Python Code')
        weakself = weakref.proxy(self)
        self.editor = self.construct_text_edit(node_property_name='code', display_property_name="python code",
                                               editor_class=PythonEditor, document=self.document)
        self.editor.setToolTip('Type Python code here. You should assign the geometry object to insert here '
                               'to the variable <tt>__object__</tt>.')
        form = self.get_current_form()
        form.addRow(self.editor)

        self._python_group = form.parent()
        self.editor.modificationChanged.connect(lambda mod: weakself._python_group.setTitle('Python Code' + ('*' if mod else '')))
        self.document.window.config_changed.connect(self.reconfig)

        self.button = QToolButton()
        self.button.setIcon(QIcon.fromTheme('view-fullscreen'))
        self.button.setCheckable(True)
        self.button.toggled.connect(self.toggle_window)
        self.button.setToolTip("Show large editor.")
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.button)
        form.addRow(row)
        self.window = None

        super().construct_form()

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.editor):
            self.editor.setPlainText(self.node.code)
        self.reconfig()

    def toggle_window(self, show):
        if show:
            if self.window is None:
                weakself = weakref.proxy(self)
                self.window = _PythonEditorWidow(weakself)
                self.window.setWindowTitle(WINDOW_TITLE)
                widget = PythonEditorWidget(self.window, self.document, line_numbers=False)
                widget.editor.setDocument(self.editor.document())
                self.editor.modificationChanged.connect(
                    lambda mod: self.window.setWindowTitle(WINDOW_TITLE + ('*' if mod else '')))
                self.document.window.config_changed.connect(self.reconfig)
                self.window.setCentralWidget(widget)
                screen = self.document.window.screen().availableGeometry()
                w, h = screen.width(), screen.height()
                save_action = QAction()
                CONFIG.set_shortcut(save_action, 'save_file')
                save_action.triggered.connect(self.window_save)
                widget.addAction(save_action)
                self.window.resize(w * 3 // 4, h * 3 // 4)
                self.window.move(w // 8, h// 8)
            self.window.show()
            self.window.raise_()
        elif self.window is not None:
            self.window.close()

    def window_save(self):
        self.window.commit()
        # self.document.window.save()

    def reconfig(self):
        self.editor.rehighlight(self.document.defines, geometry_object=['__object__'])
        if self.window is not None:
            self.window.centralWidget().editor.rehighlight(self.document.defines, geometry_object=['__object__'])
