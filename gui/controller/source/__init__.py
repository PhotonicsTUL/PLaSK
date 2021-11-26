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

from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from .. import Controller
from ...utils.config import CONFIG, parse_highlight, dark_style
from ...utils.qsignals import BlockQtSignals
from ...utils.texteditor import TextEditor, EditorWidget
from ...utils.widgets import EDITOR_FONT, set_icon_size
from ...lib.highlighter import SyntaxHighlighter, load_syntax
from ...lib.highlighter.xml import syntax
from .xml_formatter import indent, unindent, indent_new_line, parse_slash

SCHEME = {}


def update_xml_scheme():
    global SCHEME
    SCHEME = {
        'syntax_comment': parse_highlight(CONFIG['syntax/xml_comment']),
        'syntax_tag': parse_highlight(CONFIG['syntax/xml_tag']),
        'syntax_attr': parse_highlight(CONFIG['syntax/xml_attr']),
        'syntax_value': parse_highlight(CONFIG['syntax/xml_value']),
        'syntax_text': parse_highlight(CONFIG['syntax/xml_text']),
        'syntax_define': parse_highlight(CONFIG['syntax/xml_define']),
    }
update_xml_scheme()


class XMLEditor(TextEditor):

    def __init__(self, parent=None, line_numbers=True):
        super().__init__(parent, line_numbers)
        palette = self.palette()
        color = parse_highlight(CONFIG['syntax/xml_text']).get('color')
        if color is None: color = CONFIG['editor/foreground_color']
        palette.setColor(QPalette.ColorRole.Text, QColor(color))
        self.setPalette(palette)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab, Qt.Key.Key_Backspace):
            cursor = self.textCursor()
            if cursor.hasSelection():
                if key == Qt.Key.Key_Tab:
                    indent(self)
                    event.ignore()
                    return
                elif key == Qt.Key.Key_Backtab:
                    unindent(self)
                    event.ignore()
                    return
            elif key == Qt.Key.Key_Backtab:
                unindent(self)
                event.ignore()
                return
            else:
                col = cursor.positionInBlock()
                inindent = not cursor.block().text()[:col].strip()
                if inindent:
                    if key == Qt.Key.Key_Tab:
                        indent(self, col)
                        event.ignore()
                        return
                    else:
                        if not (cursor.atBlockStart()):
                            unindent(self, col)
                            event.ignore()
                            return
        elif key == Qt.Key.Key_Home and not modifiers & ~Qt.KeyboardModifier.ShiftModifier:
            cursor = self.textCursor()
            txt = cursor.block().text()
            col = cursor.positionInBlock()
            mode = QTextCursor.MoveMode.KeepAnchor if modifiers & Qt.KeyboardModifier.ShiftModifier else QTextCursor.MoveMode.MoveAnchor
            if txt[:col].strip():
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, mode)
                while self.document().characterAt(cursor.position()) in [' ', '\t']:
                    cursor.movePosition(QTextCursor.MoveOperation.Right, mode)
                self.setTextCursor(cursor)
                event.ignore()
                return
        elif key == Qt.Key.Key_Slash:
            if parse_slash(self):
                event.ignore()
                return
        elif key == Qt.Key.Key_Less:
            cursor = self.textCursor()
            if cursor.hasSelection():
                pos, anchor = cursor.position() + 1, cursor.anchor() + 1
                text = '<' + cursor.selectedText() + '>'
                cursor.insertText(text)
                cursor.setPosition(anchor)
                cursor.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)
                self.setTextCursor(cursor)
                event.ignore()
                return

        super().keyPressEvent(event)

        if key in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            indent_new_line(self)

    def reconfig(self):
        self.setFont(EDITOR_FONT)
        color = parse_highlight(CONFIG['syntax/xml_text']).get('color')
        if color is None: color = CONFIG['editor/foreground_color']
        self.setStyleSheet("QPlainTextEdit {{ color: {fg}; background-color: {bg} }}".format(
            fg=color, bg=CONFIG['editor/background_color']
        ))
        self.line_numbers.setFont(EDITOR_FONT)


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
        self.highlighter = SyntaxHighlighter(source.editor.document(),
                                             *load_syntax(syntax, SCHEME),
                                             default_font=EDITOR_FONT)
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
        editor = self.source_widget.editor
        editor.reconfig()
        if self.highlighter is not None:
            with BlockQtSignals(editor):
                update_xml_scheme()
                self.highlighter = SyntaxHighlighter(editor.document(),
                                                     *load_syntax(syntax, SCHEME),
                                                     default_font=EDITOR_FONT)

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
