from ..qt import QtGui

from ..qhighlighter.XML import XMLHighlighter
from ..utils.gui import DEFAULT_FONT
from .base import Controller

class SourceEditController(Controller):

    def __init__(self, document=None, model=None):
        Controller.__init__(self, document, model)
        self.fresh = False
        self.visible = False
        self.edited = False # True only if text has been edited after last save_data_in_model

    def create_source_editor(self, parent=None):
        ed = QtGui.QTextEdit(parent)
        ed.setFont(DEFAULT_FONT)
        self.highlighter = XMLHighlighter(ed.document())   # highlighter variable is required,
                                                           # in other case it is deleted and text is not highlighted
        ed.setReadOnly(self.model.is_read_only())
        return ed

    def __on_text_edit__(self):
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
            self.get_source_editor().setPlainText(self.model.get_text())
            self.fresh = True
        else:
            self.fresh = False

    def save_data_in_model(self):
        if not self.get_source_editor().isReadOnly() and self.edited:
            if hasattr(self.model, 'changed'): self.model.changed -= self.refresh_editor
            try:
                self.model.set_text(self.get_source_editor().toPlainText())
            finally:
                if hasattr(self.model, 'changed'): self.model.changed += self.refresh_editor
                self.edited = False

    def on_edit_enter(self):
        self.visible = True
        if not self.fresh: self.refresh_editor()
        if hasattr(self.model, 'changed'): self.model.changed += self.refresh_editor
        self.source_editor.textChanged.connect(self.__on_text_edit__)

    # When the editor is turned off, model should be updated
    def on_edit_exit(self):
        try:
            self.source_editor.textChanged.disconnect(self.__on_text_edit__)
        except TypeError:
            pass
        self.save_data_in_model()
        #if hasattr(self.model, 'changed'): self.model.changed -= self.refresh_editor
        self.visible = False
