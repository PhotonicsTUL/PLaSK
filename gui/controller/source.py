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

from . import Controller
from ..utils.config import CONFIG, parse_highlight
from ..utils.textedit import TextEdit
from ..utils.widgets import DEFAULT_FONT

from ..external.highlighter import SyntaxHighlighter, load_syntax
from ..external.highlighter.xml import syntax

scheme = {
    'syntax_comment': parse_highlight(CONFIG('syntax/xml_comment', 'color=green, italic=true')),
    'syntax_tag': parse_highlight(CONFIG('syntax/xml_tag', 'color=maroon, bold=true')),
    'syntax_attr': parse_highlight(CONFIG('syntax/xml_attr', 'color=#888800')),
    'syntax_value': parse_highlight(CONFIG('syntax/xml_value', 'color=darkblue')),
    'syntax_text': parse_highlight(CONFIG('syntax/xml_text', 'color=black')),
}


class SourceEditController(Controller):

    def __init__(self, document=None, model=None):
        Controller.__init__(self, document, model)
        self.fresh = False
        self.visible = False
        self.edited = False  # True only if text has been edited after last save_data_in_model
        self.source_editor = None

    def create_source_editor(self, parent=None):
        edit = TextEdit(parent)
        edit.setFont(DEFAULT_FONT)
        self.highlighter = SyntaxHighlighter(edit.document(), *load_syntax(syntax, scheme), default_font=DEFAULT_FONT)
        edit.setReadOnly(self.model.is_read_only())
        return edit

    def _on_text_edit(self):
        self.edited = True
        self.document.set_changed()

    # text, source editor
    def get_source_editor(self):
        if self.source_editor is None:
            self.source_editor = self.create_source_editor(self.document.window)
        return self.source_editor

    # GUI editor, by default use source editor
    def get_editor(self):
        return self.get_source_editor()

    def refresh_editor(self, *args, **kwargs):
        if self.visible:
            editor = self.get_source_editor()
            editor.setPlainText(self.model.get_text())
            self.fresh = True
        else:
            self.fresh = False

    def save_data_in_model(self):
        if not self.get_source_editor().isReadOnly() and self.edited:
            try: self.model.changed -= self.refresh_editor
            except AttributeError: pass
            try:
                self.model.set_text(self.get_source_editor().toPlainText())
                self.edited = False
            finally:
                try: self.model.changed += self.refresh_editor
                except AttributeError: pass

    def on_edit_enter(self):
        self.visible = True
        if not self.fresh: self.refresh_editor()
        try: self.source_editor.line_numbers.offset = self.model.line_in_file
        except AttributeError: pass
        try: self.model.changed += self.refresh_editor
        except AttributeError: pass
        self.document.window.showsource_action.setChecked(True)
        self.source_editor.textChanged.connect(self._on_text_edit)

    # When the editor is turned off, model should be updated
    def on_edit_exit(self):
        try:
            self.source_editor.textChanged.disconnect(self._on_text_edit)
        except:
            pass
        self.save_data_in_model()
        #if hasattr(self.model, 'changed'): self.model.changed -= self.refresh_editor
        self.visible = False
        self.document.window.showsource_action.setChecked(False)
