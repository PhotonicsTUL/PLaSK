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

from ...qt import QtCore, QtGui
from ...qt.QtCore import Qt

from .. import Controller
from ...utils.config import CONFIG, parse_highlight
from ...utils.textedit import TextEdit
from ...utils.widgets import DEFAULT_FONT

from ...external.highlighter import SyntaxHighlighter, load_syntax
from ...external.highlighter.xml import syntax

from .indenter import indent, unindent, autoindent


scheme = {
    'syntax_comment': parse_highlight(CONFIG('syntax/xml_comment', 'color=green, italic=true')),
    'syntax_tag': parse_highlight(CONFIG('syntax/xml_tag', 'color=maroon, bold=true')),
    'syntax_attr': parse_highlight(CONFIG('syntax/xml_attr', 'color=#888800')),
    'syntax_value': parse_highlight(CONFIG('syntax/xml_value', 'color=darkblue')),
    'syntax_text': parse_highlight(CONFIG('syntax/xml_text', 'color=black')),
}


class XMLEditor(TextEdit):

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Backspace):
            cursor = self.textCursor()
            if cursor.hasSelection():
                if key == Qt.Key_Tab:
                    indent(self)
                    return
                elif key == Qt.Key_Backtab:
                    unindent(self)
                    return
            elif key == Qt.Key_Backtab:
                unindent(self)
                return
            else:
                col = cursor.positionInBlock()
                inindent = not cursor.block().text()[:col].strip()
                if inindent:
                    if key == Qt.Key_Tab:
                        indent(self, col)
                        return
                    else:
                        if not (cursor.atBlockStart()):
                            unindent(self, col)
                            return
        elif key == Qt.Key_Home and not modifiers & ~Qt.ShiftModifier:
            cursor = self.textCursor()
            txt = cursor.block().text()
            col = cursor.positionInBlock()
            mode = QtGui.QTextCursor.KeepAnchor if modifiers & Qt.ShiftModifier else QtGui.QTextCursor.MoveAnchor
            if txt[:col].strip():
                cursor.movePosition(QtGui.QTextCursor.StartOfBlock, mode)
                while self.document().characterAt(cursor.position()) in [' ', '\t']:
                    cursor.movePosition(QtGui.QTextCursor.Right, mode)
                self.setTextCursor(cursor)
                return

        super(XMLEditor, self).keyPressEvent(event)

        if key in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Colon):
            autoindent(self)


class SourceWidget(QtGui.QWidget):

    def __init__(self, parent=None, editor_class=TextEdit, *args, **kwargs):
        super(SourceWidget, self).__init__(parent)

        self.editor = editor_class(self, *args, **kwargs)
        self.editor.setFont(DEFAULT_FONT)

        self.toolbar = QtGui.QToolBar(self)
        self.toolbar.setStyleSheet("QToolBar { border: 0px }")

        self.add_action('&Undo', 'edit-undo', None, self.editor.undo)
        self.add_action('R&edo', 'edit-redo', None, self.editor.redo)
        self.toolbar.addSeparator()
        self.add_action('&Copy', 'edit-copy', None, self.editor.copy)
        self.add_action('C&ut', 'edit-cut', None, self.editor.cut)
        self.add_action('&Paste', 'edit-paste', None, self.editor.paste)
        self.toolbar.addSeparator()
        self.add_action('&Find...', 'edit-find', QtGui.QKeySequence.Find, self.show_find)
        self.add_action('&Replace...', 'edit-find-replace', QtGui.QKeySequence.Replace, self.show_replace)

        self.make_find_replace_widget()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.editor)
        layout.addWidget(self.find_toolbar)
        layout.addWidget(self.replace_toolbar)

        layout.setContentsMargins(0, 1, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)

    def make_find_replace_widget(self):
        self.find_toolbar = QtGui.QToolBar(self)
        self.replace_toolbar = QtGui.QToolBar(self)
        self.find_toolbar.setStyleSheet("QToolBar { border: 0px }")
        self.replace_toolbar.setStyleSheet("QToolBar { border: 0px }")
        find_label = QtGui.QLabel()
        find_label.setText("Search: ")
        find_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        replace_label = QtGui.QLabel()
        replace_label.setText("Replace: ")
        replace_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label_width = replace_label.fontMetrics().width(replace_label.text())
        find_label.setFixedWidth(label_width)
        replace_label.setFixedWidth(label_width)
        self.find_edit = QtGui.QLineEdit()
        self.find_toolbar.addWidget(find_label)
        self.find_toolbar.addWidget(self.find_edit)
        self.replace_edit = QtGui.QLineEdit()
        self.replace_toolbar.addWidget(replace_label)
        self.replace_toolbar.addWidget(self.replace_edit)

        self.find_matchcase = QtGui.QAction('&Match Case', self.find_edit)
        self.find_matchcase.setCheckable(True)
        self.find_matchcase.setChecked(True)
        self.find_wholewords = QtGui.QAction('&Whole Words', self.find_edit)
        self.find_wholewords.setCheckable(True)
        self.find_regex = QtGui.QAction('&Regular Expression', self.find_edit)
        self.find_regex.setCheckable(True)
        self.find_edit.setContextMenuPolicy(Qt.CustomContextMenu)
        self.find_edit.customContextMenuRequested.connect(self._find_context_menu)
        self.find_options = QtGui.QMenu()
        self.find_options.addAction(self.find_matchcase)
        self.find_options.addAction(self.find_wholewords)
        self.find_options.addAction(self.find_regex)
        options_button = QtGui.QPushButton(self)
        options_button.setText("Opt&ions")
        options_button.setMenu(self.find_options)
        self.find_toolbar.addWidget(options_button)

        next_button = QtGui.QPushButton(self)
        next_button.setText("&Next")
        next_button.setFixedWidth(150)  # TODO from maximum text+icon width
        next_button.pressed.connect(self.find_next)
        prev_button = QtGui.QPushButton(self)
        prev_button.setText("&Previous")  # TODO from maximum text+icon width
        prev_button.setFixedWidth(150)
        prev_button.pressed.connect(self.find_prev)
        self.find_toolbar.addWidget(next_button)
        self.find_toolbar.addWidget(prev_button)
        replace_button = QtGui.QPushButton(self)
        replace_button.setText("Rep&lace one")
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
        self._add_shortcut(QtGui.QKeySequence(Qt.Key_Escape), self.hide_toolbars)
        self._add_shortcut(QtGui.QKeySequence.FindNext, self.find_next)
        self._add_shortcut(QtGui.QKeySequence.FindPrevious, self.find_prev)
        self.find_edit.textEdited.connect(self.find_type)
        self.find_edit.returnPressed.connect(self.find_next)
        self.replace_edit.returnPressed.connect(self.replace_next)

    def _find_context_menu(self, pos):
        menu = self.find_edit.createStandardContextMenu()
        menu.addSeparator()
        menu.addAction(self.find_matchcase)
        menu.addAction(self.find_wholewords)
        menu.addAction(self.find_regex)
        menu.exec_(self.find_toolbar.mapToGlobal(pos))

    def _find_flags(self):
        return \
            (QtGui.QTextDocument.FindCaseSensitively if self.find_matchcase.isChecked() else 0) | \
            (QtGui.QTextDocument.FindWholeWords if self.find_wholewords.isChecked() else 0)

    def add_action(self, name, icon, shortcut, slot):
        action = QtGui.QAction(QtGui.QIcon.fromTheme(icon), name, self)
        if shortcut is not None:
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
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            self.find_edit.setText(cursor.selectedText())
        self.find_edit.selectAll()
        self.find_edit.setPalette(self.editor.palette())
        self.find_toolbar.show()
        self.find_edit.setFocus()

    def show_replace(self):
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            self.find_edit.setText(cursor.selectedText())
        self.find_edit.selectAll()
        self.find_edit.setPalette(self.editor.palette())
        self.find_toolbar.show()
        self.replace_toolbar.show()
        self.find_edit.setFocus()

    def hide_toolbars(self):
        self.find_toolbar.hide()
        self.replace_toolbar.hide()
        self.editor.setFocus()

    def _find(self, cont=False, backward=False, rewind=True):
        cursor = self.editor.textCursor()
        if cont:
            cursor.setPosition(cursor.selectionStart())
        pal = self.editor.palette()
        if self.find_regex.isChecked():
            self._findtext = QtCore.QRegExp(self.find_edit.text())
        else:
            self._findtext = self.find_edit.text()
        if self._findtext:
            document = self.editor.document()
            findflags = self._find_flags() | (QtGui.QTextDocument.FindBackward if backward else 0)
            found = document.find(self._findtext, cursor, findflags)
            if found.isNull() and rewind:
                cursor.movePosition(QtGui.QTextCursor.End if backward else QtGui.QTextCursor.Start)
                found = document.find(self._findtext, cursor, findflags)
            if found.isNull():
                pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#fdd"))
                self.find_edit.setPalette(pal)
                return False
            else:
                self.editor.setTextCursor(found)
                pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#dfd"))
                self.find_edit.setPalette(pal)
                return True
        else:
            self.find_edit.setPalette(pal)
            cursor.setPosition(cursor.position())
            self.editor.setTextCursor(cursor)

    def find_next(self):
        self._find()
        #self.editor.setFocus()

    def find_prev(self):
        self._find(backward=True)
        #self.editor.setFocus()

    def find_type(self):
        self._find(cont=True)

    def replace_next(self, rewind=True):
        if not self._find(cont=True, rewind=rewind):
            return False
        pal = self.editor.palette()
        self.find_edit.setPalette(pal)
        cursor = self.editor.textCursor()
        start = cursor.selectionStart()
        if isinstance(self._findtext, QtCore.QRegExp):
            cursor.insertText(self._findtext.replace(cursor.selectedText(), self.replace_edit.text()))
        else:
            cursor.insertText(self.replace_edit.text())
        end = cursor.position()
        if not self._find(cont=False, rewind=rewind):
            cursor.setPosition(start)
            cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
            self.editor.setTextCursor(cursor)
            return False
        # self.editor.setFocus()
        return True

    def replace_all(self):
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        try:
            cursor.movePosition(QtGui.QTextCursor.Start)
            self.editor.setTextCursor(cursor)
            while self.replace_next(rewind=False): pass
        finally:
            cursor.endEditBlock()


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
        source = SourceWidget(parent, XMLEditor, line_numbers=self.line_numbers)
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
            text = self.model.get_text()
            if text and text[-1] == '\n':
                text = text[:-1]
            editor.setPlainText(text)
            self.fresh = True
        else:
            self.fresh = False

    def save_data_in_model(self):
        if not self.get_source_widget().editor.isReadOnly() and self.edited:
            try: self.model.changed -= self.refresh_editor
            except AttributeError: pass
            try:
                self.model.set_text(self.get_source_widget().editor.toPlainText() + '\n')
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