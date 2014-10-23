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

import sys

from ...qt import QtCore, QtGui
from ...qt.QtCore import Qt

from ...model.script.completer import CompletionsModel, get_completions

from .brackets import get_selections as get_bracket_selections
from .indenter import indent, unindent, autoindent

from ..source import SourceEditController, SourceWidget
from ...model.script import ScriptModel
from ...utils.config import CONFIG, parse_highlight
from ...utils.widgets import DEFAULT_FONT
from ...utils.textedit import TextEdit

from ...external.highlighter import SyntaxHighlighter, load_syntax
if sys.version_info >= (3, 0, 0):
    from ...external.highlighter.python32 import syntax
else:
    from ...external.highlighter.python27 import syntax
from ...external.highlighter.plask import syntax as plask_syntax

syntax['formats'].update(plask_syntax['formats'])
syntax['scanner'][None] = syntax['scanner'][None][:-1] + plask_syntax['scanner'] + [syntax['scanner'][None][-1]]

scheme = {
    'syntax_comment': parse_highlight(CONFIG('syntax/python_comment', 'color=green, italic=true')),
    'syntax_string': parse_highlight(CONFIG('syntax/python_string', 'color=blue')),
    'syntax_builtin': parse_highlight(CONFIG('syntax/python_builtin', 'color=maroon')),
    'syntax_keyword': parse_highlight(CONFIG('syntax/python_keyword', 'color=black, bold=true')),
    'syntax_number': parse_highlight(CONFIG('syntax/python_number', 'color=darkblue')),
    'syntax_member': parse_highlight(CONFIG('syntax/python_member', 'color=#440044')),
    'syntax_plask': parse_highlight(CONFIG('syntax/python_plask', 'color=#0088ff')),
    'syntax_provider': parse_highlight(CONFIG('syntax/python_provider', 'color=#888800')),
    'syntax_receiver': parse_highlight(CONFIG('syntax/python_receiver', 'color=#888800')),
    'syntax_log': parse_highlight(CONFIG('syntax/python_log', 'color=blue')),
    'syntax_solver': parse_highlight(CONFIG('syntax/python_solver', 'color=red')),
    'syntax_loaded': parse_highlight(CONFIG('syntax/python_loaded', 'color=#ff8800')),
    'syntax_pylab': parse_highlight(CONFIG('syntax/python_pylab', 'color=#880044')),
}


class ScriptEditor(TextEdit):
    """Editor with some features usefult for script editing"""

    def __init__(self, parent=None, controller=None):
        self.controller = controller
        super(ScriptEditor, self).__init__(parent)

        self.completer = QtGui.QCompleter()
        self.completer.setWidget(self)
        self.completer.setCompletionMode(QtGui.QCompleter.PopupCompletion)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.activated.connect(self.insert_completion)

        self.cursorPositionChanged.connect(self.update_selections)

        self.comment_action = QtGui.QAction('Co&mment lines', self)
        self.uncomment_action = QtGui.QAction('Uncomm&ent lines', self)
        self.comment_action.setShortcut(Qt.CTRL + Qt.Key_Slash)
        self.uncomment_action.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_Slash)
        self.comment_action.triggered.connect(self.block_comment)
        self.uncomment_action.triggered.connect(self.block_uncomment)
        self.addAction(self.comment_action)
        self.addAction(self.uncomment_action)

    def update_selections(self):
        """Add our own custom selections"""
        col = self.textCursor().positionInBlock()
        brackets = get_bracket_selections(self, self.textCursor().block(), col)
        self.setExtraSelections(self.extraSelections() + brackets)

    def block_comment(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
        if cursor.position() == end: end += 1
        while cursor.position() < end:
            cursor.insertText("# ")
            end += 2
            if not cursor.movePosition(QtGui.QTextCursor.NextBlock):
                break
        cursor.endEditBlock()

    def block_uncomment(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
        if cursor.position() == end: end += 1
        document = self.document()
        try:
            while cursor.position() < end:
                while document.characterAt(cursor.position()) in (' ', '\t'):
                    if not cursor.movePosition(QtGui.QTextCursor.NextCharacter): raise ValueError
                if document.characterAt(cursor.position()) == '#':
                    cursor.deleteChar()
                    end -= 1
                    if document.characterAt(cursor.position()) == ' ':
                        cursor.deleteChar()
                        end -= 1
                if not cursor.movePosition(QtGui.QTextCursor.NextBlock): raise ValueError
        except ValueError:
            pass
        cursor.endEditBlock()

    def insert_completion(self, completion):
        # if self.completer.widget() != self: return
        cursor = self.textCursor()
        extra = len(self.completer.completionPrefix())
        cursor.movePosition(QtGui.QTextCursor.Left)
        cursor.movePosition(QtGui.QTextCursor.EndOfWord)
        cursor.insertText(completion[extra:])
        self.setTextCursor(cursor)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if self.completer.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape, Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return  # let the completer do default behaviors
            if event.text():
                self.completer.setCompletionPrefix(self.completer.completionPrefix() + event.text())

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

        if key != Qt.Key_Period or modifiers != Qt.ControlModifier:
            super(ScriptEditor, self).keyPressEvent(event)

        eow = "~!@#$%^&*()_+{}|:\"<>?,./;'[]\\-="  # end of word
        has_modifier = modifiers != Qt.NoModifier and not modifiers & (Qt.ControlModifier | Qt.ShiftModifier)
        if has_modifier or not event.text() or event.text()[-1] in eow:
            self.completer.popup().hide()

        if key in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Colon):
            autoindent(self)
        elif key == Qt.Key_Period and modifiers in (Qt.NoModifier, Qt.ControlModifier):
            cursor = self.textCursor()
            row = cursor.blockNumber()
            col = cursor.positionInBlock()
            cursor.select(QtGui.QTextCursor.WordUnderCursor)
            completion_prefix = cursor.selectedText()
            items = get_completions(self.toPlainText(), row, col)
            if items:
                self.completer.setModel(QtGui.QStringListModel(items, self.completer))
                if completion_prefix != self.completer.completionPrefix():
                    self.completer.setCompletionPrefix(completion_prefix)
                    self.completer.popup().setCurrentIndex(self.completer.completionModel().index(0, 0))
                cr = self.cursorRect()
                cr.setWidth(self.completer.popup().sizeHintForColumn(0)
                            + self.completer.popup().verticalScrollBar().sizeHint().width())
                self.completer.complete(cr)  # popup it up!`


class ScriptController(SourceEditController):

    def __init__(self, document, model=None):
        if model is None: model = ScriptModel()
        SourceEditController.__init__(self, document, model)

    def create_source_widget(self, parent):
        source = SourceWidget(parent, ScriptEditor, self)
        source.editor.setReadOnly(self.model.is_read_only())

        source.toolbar.addSeparator()
        menu = QtGui.QMenu()
        menu.addAction(source.editor.comment_action)
        menu.addAction(source.editor.uncomment_action)
        button = QtGui.QToolButton()
        button.setIcon(QtGui.QIcon.fromTheme('code-block', QtGui.QIcon(':/code-block')))
        button.setMenu(menu)
        button.setPopupMode(QtGui.QToolButton.InstantPopup)
        source.toolbar.addWidget(button)
        if self.model.is_read_only():
            source.editor.comment_action.setEnabled(False)
            source.editor.uncomment_action.setEnabled(False)

        self.highlighter = SyntaxHighlighter(source.editor.document(),
                                             *load_syntax(syntax, scheme),
                                             default_font=DEFAULT_FONT)

        return source

    def on_edit_enter(self):
        super(ScriptController, self).on_edit_enter()

    def on_edit_exit(self):
        return super(ScriptController, self).on_edit_exit()
