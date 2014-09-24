from ..qt import QtGui

from .base import Controller
from ..utils.config import CONFIG, parse_highlight
from ..utils.textedit import TextEdit
from ..utils.gui import DEFAULT_FONT

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
        self.edited = False # True only if text has been edited after last save_data_in_model

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
        if not hasattr(self, 'source_editor'): self.source_editor = self.create_source_editor(self.document.window)
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
            if hasattr(self.model, 'changed'): self.model.changed -= self.refresh_editor
            try:
                self.model.set_text(self.get_source_editor().toPlainText())
                self.edited = False
            finally:
                if hasattr(self.model, 'changed'): self.model.changed += self.refresh_editor

    def on_edit_enter(self):
        self.visible = True
        if not self.fresh: self.refresh_editor()
        try: self.source_editor.line_numbers.offset = self.model.line_in_file
        except AttributeError: pass
        if hasattr(self.model, 'changed'): self.model.changed += self.refresh_editor
        self.source_editor.textChanged.connect(self._on_text_edit)

    # When the editor is turned off, model should be updated
    def on_edit_exit(self):
        try:
            self.source_editor.textChanged.disconnect(self._on_text_edit)
        except TypeError:
            pass
        except AttributeError:
            pass
        self.save_data_in_model()
        #if hasattr(self.model, 'changed'): self.model.changed -= self.refresh_editor
        self.visible = False
