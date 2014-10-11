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

from ..qt import QtCore, QtGui

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

    def __init__(self, document=None, model=None, line_numbers=True):
        Controller.__init__(self, document, model)
        self.line_numbers = line_numbers
        self.fresh = False
        self.visible = False
        self.edited = False  # True only if text has been edited after last save_data_in_model
        self.source_widget = None

    def _on_text_edit(self):
        self.edited = True
        self.document.set_changed()

    def create_source_widget(self, parent):
        source = SourceWidget(parent, line_numbers=self.line_numbers)
        self.highlighter = SyntaxHighlighter(source.editor.document(),
                                             *load_syntax(syntax, scheme),
                                             default_font=DEFAULT_FONT)
        source.editor.setReadOnly(self.model.is_read_only())
        return source

    def get_source_widget(self):
        if self.source_widget is None:
            self.source_widget = self.create_source_widget(self.document.window)
        return self.source_widget

    # GUI editor, by default use source editor
    def get_widget(self):
        return self.get_source_widget()

    def refresh_editor(self, *args, **kwargs):
        if self.visible:
            editor = self.get_source_widget().editor
            editor.setPlainText(self.model.get_text())
            self.fresh = True
        else:
            self.fresh = False

    def save_data_in_model(self):
        if not self.get_source_widget().editor.isReadOnly() and self.edited:
            try: self.model.changed -= self.refresh_editor
            except AttributeError: pass
            try:
                self.model.set_text(self.get_source_widget().editor.toPlainText())
                self.edited = False
            finally:
                try: self.model.changed += self.refresh_editor
                except AttributeError: pass

    def on_edit_enter(self):
        self.visible = True
        if not self.fresh: self.refresh_editor()
        try: self.source_widget.editor.line_numbers.offset = self.model.line_in_file
        except AttributeError: pass
        try: self.model.changed += self.refresh_editor
        except AttributeError: pass
        self.source_widget.editor.textChanged.connect(self._on_text_edit)

    # When the editor is turned off, the model should be updated
    def on_edit_exit(self):
        if not self.try_save_data_in_model():
            return False
        self.source_widget.editor.textChanged.disconnect(self._on_text_edit)
        #if hasattr(self.model, 'changed'): self.model.changed -= self.refresh_editor
        self.visible = False
        return True


class SourceWidget(QtGui.QWidget):

    def __init__(self, parent=None, editor_class=TextEdit, *args, **kwargs):
        super(SourceWidget, self).__init__(parent)

        self.editor = editor_class(self, *args, **kwargs)
        self.editor.setFont(DEFAULT_FONT)

        self.toolbar = QtGui.QToolBar(self)

        self.add_action('&Undo', 'edit-undo', QtGui.QKeySequence.Undo, self.editor.undo)
        self.add_action('R&edo', 'edit-redo', QtGui.QKeySequence.Redo, self.editor.redo)
        self.toolbar.addSeparator()
        self.add_action('&Copy', 'edit-copy', QtGui.QKeySequence.Copy, self.editor.copy)
        self.add_action('C&ut', 'edit-cut', QtGui.QKeySequence.Copy, self.editor.cut)
        self.add_action('&Paste', 'edit-paste', QtGui.QKeySequence.Copy, self.editor.paste)
        self.toolbar.addSeparator()
        self.add_action('&Find...', 'edit-find', QtGui.QKeySequence.Find, self.show_find)
        self.add_action('&Replace...', 'edit-find-replace', QtGui.QKeySequence.Replace, self.show_replace)

        self.make_find_replace_widget()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.editor)
        layout.addWidget(self.find_toolbar)
        layout.addWidget(self.replace_toolbar)
        self.setLayout(layout)

    def make_find_replace_widget(self):
        self.find_toolbar = QtGui.QToolBar(self)
        self.replace_toolbar = QtGui.QToolBar(self)
        find_label = QtGui.QLabel()
        find_label.setText("Search:")
        find_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        replace_label = QtGui.QLabel()
        replace_label.setText("Replace:")
        replace_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label_width = replace_label.fontMetrics().width(replace_label.text())
        find_label.setFixedWidth(label_width)
        replace_label.setFixedWidth(label_width)
        self.find_edit = QtGui.QLineEdit()
        self.find_toolbar.addWidget(find_label)
        self.find_toolbar.addWidget(self.find_edit)
        self.replace_edit = QtGui.QLineEdit()
        self.replace_toolbar.addWidget(replace_label)
        self.replace_toolbar.addWidget(self.replace_edit)
        next_button = QtGui.QPushButton(self)
        next_button.setText("&Next")
        next_button.setDefault(True)
        next_button.setFixedWidth(150)  # TODO from maximum text+icon width
        next_button.pressed.connect(self.find_next)
        prev_button = QtGui.QPushButton(self)
        prev_button.setText("&Previous")  # TODO from maximum text+icon width
        prev_button.setFixedWidth(150)
        prev_button.pressed.connect(self.find_prev)
        self.find_toolbar.addWidget(next_button)
        self.find_toolbar.addWidget(prev_button)
        replace_button = QtGui.QPushButton(self)
        replace_button.setText("R&eplace one")
        replace_button.setFixedWidth(150)  # TODO from maximum text+icon width
        replace_button.pressed.connect(self.replace_next)
        replace_all_button = QtGui.QPushButton(self)
        replace_all_button.setText("Replace &all")  # TODO from maximum text+icon width
        replace_all_button.setFixedWidth(150)
        replace_all_button.pressed.connect(self.replace_all)
        self.replace_toolbar.addWidget(replace_button)
        self.replace_toolbar.addWidget(replace_all_button)
        self.find_toolbar.hide()
        self.replace_toolbar.hide()
        self._add_shortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self.hide_toolbars)
        self._add_shortcut(QtGui.QKeySequence.FindNext, self.find_next)
        self._add_shortcut(QtGui.QKeySequence.FindPrevious, self.find_prev)

    def add_action(self, name, icon, shortcut, slot):
        action = QtGui.QAction(QtGui.QIcon.fromTheme(icon, QtGui.QIcon(':/{}'.format(icon))), name, self)
        action.setShortcut(shortcut)
        action.triggered.connect(slot)
        self.toolbar.addAction(action)
        return action

    def _add_shortcut(self, shortcut, slot):
        action = QtGui.QAction(self)
        action.setShortcut(shortcut)
        action.triggered.connect(slot)
        self.editor.addAction(action)
        return action

    def show_find(self):
        self.find_toolbar.show()
        self.find_edit.setFocus()

    def show_replace(self):
        self.find_toolbar.show()
        self.replace_toolbar.show()
        self.find_edit.setFocus()

    def hide_toolbars(self):
        self.find_toolbar.hide()
        self.replace_toolbar.hide()
        self.editor.setFocus()

    def find_next(self):
        self.editor.find(self.find_edit.text(), QtGui.QTextDocument.FindCaseSensitively)

    def find_prev(self):
        self.editor.find(self.find_edit.text(),
                         QtGui.QTextDocument.FindCaseSensitively | QtGui.QTextDocument.FindBackward)

    def replace_next(self):
        if not self.editor.find(self.find_edit.text(), QtGui.QTextDocument.FindCaseSensitively):
            return False
        cursor = self.editor.textCursor()
        start = cursor.selectionStart()
        cursor.insertText(self.replace_edit.text())
        end = cursor.position()
        cursor.setPosition(start)
        cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
        self.editor.setTextCursor(cursor)
        self.editor.setFocus()
        return True

    def replace_all(self):
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        try:
            cursor.movePosition(QtGui.QTextCursor.Start)
            self.editor.setTextCursor(cursor)
            while self.replace_next(): pass
        finally:
            cursor.endEditBlock()

    def get_widget(self):
        return self.editor