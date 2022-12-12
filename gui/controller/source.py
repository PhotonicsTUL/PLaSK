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


from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from . import Controller
from ..utils.qsignals import BlockQtSignals
from ..utils.texteditor import EditorWidget
from ..utils.texteditor.xml import XMLEditor


class SourceEditController(Controller):

    def __init__(self, document=None, model=None, line_numbers=True):
        Controller.__init__(self, document, model)
        self.line_numbers = line_numbers
        self._dirty = True
        self._visible = False
        self.source_widget = None
        self.document.window.config_changed.connect(self.reconfig)
        self.highlighter = None

    def create_source_widget(self, parent):
        source = EditorWidget(parent, XMLEditor, line_numbers=self.line_numbers)
        source.editor.setReadOnly(self.model.is_read_only())
        return source

    def get_source_widget(self):
        if self.source_widget is None:
            self.source_widget = self.create_source_widget(self.document.window)
        return self.source_widget

    # GUI editor, by default use source editor
    def get_widget(self):
        return self.get_source_widget()

    def reconfig(self):
        self.source_widget.editor.reconfig()

    def load_data_from_model(self):
        if self._dirty:
            editor = self.get_source_widget().editor
            text = self.model.get_text()
            if text and text[-1] == '\n':
                text = text[:-1]
            editor.setPlainText(text)
            self._dirty = False

    def save_data_in_model(self):
        if not self.get_source_widget().editor.isReadOnly() and \
                self.get_source_widget().editor.document().isModified():
            try: self.model.changed -= self.update_editor
            except AttributeError: pass
            try:
                self.model.set_text(self.get_source_widget().editor.toPlainText() + '\n')
            finally:
                try: self.model.changed += self.update_editor
                except AttributeError: pass
            # with BlockQtSignals(self.get_source_widget().editor.document()):
            #     self.get_source_widget().editor.document().setModified(False)

    def _modification_changed(self, changed):
            self.document.set_changed(changed)

    def update_editor(self, *args, **kwargs):
        if self._visible:
            self.load_data_from_model()
        else:
            self._dirty = True

    def on_edit_enter(self):
        self._visible = True
        self.load_data_from_model()
        try: self.source_widget.editor.line_numbers.offset = self.model.line_in_file
        except AttributeError: pass
        try: self.model.changed += self.update_editor
        except AttributeError: pass
        self.source_widget.editor.document().setModified(self.document.is_changed())
        self.source_widget.editor.modificationChanged.connect(self._modification_changed)

    # When the editor is turned off, the model should be updated
    def on_edit_exit(self):
        if not self.try_save_data_in_model():
            return False
        try:
            self.source_widget.editor.modificationChanged.disconnect(self._modification_changed)
        except (RuntimeError, TypeError):
            pass
        #if hasattr(self.model, 'changed'): self.model.changed -= self.refresh_editor
        self._visible = False
        return True
