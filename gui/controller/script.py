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

import sys
import weakref

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from .source import SourceEditController
from ..model.script import ScriptModel
from ..utils.config import CONFIG
from ..utils.widgets import ComboBox
from ..utils.texteditor.python import PythonEditorWidget, PYTHON_SCHEME


LOG_LEVELS = ['Error', 'Warning', 'Important', 'Info', 'Result', 'Data', 'Detail', 'Debug']


class ScriptController(SourceEditController):

    def __init__(self, document, model=None):
        if model is None: model = ScriptModel()
        super().__init__(document, model)
        self.document.window.config_changed.connect(self.reconfig)

    def create_source_widget(self, parent):
        window = PythonEditorWidget(parent, self.document, self.model.is_read_only())
        self.model.editor = window.editor

        try:
            loglevel = self.document.loglevel
        except AttributeError:
            pass
        else:
            widget = window.centralWidget()
            spacer = QWidget()
            spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            widget.toolbar.addWidget(spacer)
            widget.toolbar.addWidget(QLabel("Log Level: "))
            self.loglevel = ComboBox()
            self.loglevel.addItems(LOG_LEVELS)
            try:
                self.loglevel.setCurrentIndex(LOG_LEVELS.index(loglevel.title()))
            except ValueError:
                self.loglevel.setCurrentIndex(6)
            self.loglevel.currentIndexChanged.connect(self.document.set_loglevel)
            widget.toolbar.addWidget(self.loglevel)

        return window

    def _modification_changed(self, changed):
        self.model.undo_stack.cleanChanged.emit(changed)

    def on_edit_enter(self):
        super().on_edit_enter()
        self.source_widget.editor.rehighlight(self.document.defines, self.document.solvers)

    def reconfig(self):
        self.source_widget.editor.reconfig(self.document.defines, self.document.solvers)

    def on_edit_exit(self):
        return super().on_edit_exit()

    def save_data_in_model(self):
        self.before_save()
        super().save_data_in_model()

    def before_save(self):
        if CONFIG['editor/remove_trailing_spaces']:
            self.load_data_from_model()
            editor = self.get_source_widget().editor
            editor.remove_trailing_spaces()
            self.model._code = editor.document().toPlainText()
