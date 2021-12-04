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

from ...qt import QT_API
from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from ...utils.qsignals import BlockQtSignals
from ...utils.config import CONFIG, parse_highlight, set_font
from ...utils.widgets import EDITOR_FONT
from ...lib.highlighter import SyntaxHighlighter, load_syntax
from ...lib.highlighter.plask import get_syntax
from . import TextEditor
from .brackets import get_selections as get_bracket_selections, update_brackets_colors
from .indenter import indent, unindent, autoindent


PYTHON_SCHEME = {}


def update_python_scheme():
    global PYTHON_SCHEME
    PYTHON_SCHEME['syntax_comment'] = parse_highlight(CONFIG['syntax/python_comment'])
    PYTHON_SCHEME['syntax_string'] = parse_highlight(CONFIG['syntax/python_string'])
    PYTHON_SCHEME['syntax_special'] = parse_highlight(CONFIG['syntax/python_special'])
    PYTHON_SCHEME['syntax_builtin'] = parse_highlight(CONFIG['syntax/python_builtin'])
    PYTHON_SCHEME['syntax_keyword'] = parse_highlight(CONFIG['syntax/python_keyword'])
    PYTHON_SCHEME['syntax_number'] = parse_highlight(CONFIG['syntax/python_number'])
    PYTHON_SCHEME['syntax_decorator'] = parse_highlight(CONFIG['syntax/python_decorator'])
    PYTHON_SCHEME['syntax_member'] = parse_highlight(CONFIG['syntax/python_member'])
    PYTHON_SCHEME['syntax_plask'] = parse_highlight(CONFIG['syntax/python_plask'])
    PYTHON_SCHEME['syntax_provider'] = parse_highlight(CONFIG['syntax/python_provider'])
    PYTHON_SCHEME['syntax_receiver'] = parse_highlight(CONFIG['syntax/python_receiver'])
    PYTHON_SCHEME['syntax_log'] = parse_highlight(CONFIG['syntax/python_log'])
    PYTHON_SCHEME['syntax_solver'] = parse_highlight(CONFIG['syntax/python_solver'])
    PYTHON_SCHEME['syntax_define'] = parse_highlight(CONFIG['syntax/python_define'])
    PYTHON_SCHEME['syntax_loaded'] = parse_highlight(CONFIG['syntax/python_loaded'])
    PYTHON_SCHEME['syntax_pylab'] = parse_highlight(CONFIG['syntax/python_pylab'])
    PYTHON_SCHEME['syntax_obsolete'] = {'color': '#aaaaaa', 'bold': True, 'italic': True}
update_python_scheme()


class PythonTextEditor(TextEditor):
    """Editor with some features usefult for script editing"""

    def __init__(self, parent=None, line_numbers=True):
        super().__init__(parent, line_numbers)

        self.cursorPositionChanged.connect(self.highlight_brackets)
        self.selectionChanged.connect(self.highlight_brackets)

        self.comment_action = QAction('Co&mment Lines', self)
        self.uncomment_action = QAction('Uncomm&ent Lines', self)
        self.toggle_comment_action = QAction('&Toggle Comment on Lines', self)
        self.join_lines_action = QAction('Join Lines', self)
        CONFIG.set_shortcut(self.comment_action, 'python_comment')
        CONFIG.set_shortcut(self.uncomment_action, 'python_uncomment')
        CONFIG.set_shortcut(self.toggle_comment_action, 'python_toggle_comment')
        CONFIG.set_shortcut(self.join_lines_action, 'python_join_lines')
        self.comment_action.triggered.connect(self.block_comment)
        self.uncomment_action.triggered.connect(self.block_uncomment)
        self.toggle_comment_action.triggered.connect(self.block_comment_toggle)
        self.join_lines_action.triggered.connect(self.join_lines)
        self.addAction(self.comment_action)
        self.addAction(self.uncomment_action)
        self.addAction(self.toggle_comment_action)
        self.addAction(self.join_lines_action)

    def highlight_brackets(self):
        self.setExtraSelections(self.extraSelections() +
                                get_bracket_selections(self, self.textCursor().block(),
                                                       self.textCursor().positionInBlock()))

    def block_comment(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        start = cursor.position()
        if start == end: end += 1
        document = self.document()
        margin = inf
        while cursor.position() < end:
            while document.characterAt(cursor.position()) in (' ', '\t'):
                if not cursor.movePosition(QTextCursor.MoveOperation.NextCharacter): break
            margin = min(cursor.positionInBlock(), margin)
            if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock):
                break
        cursor.setPosition(start)
        while cursor.position() < end:
            cursor.movePosition(QTextCursor.MoveOperation.NextCharacter, QTextCursor.MoveMode.MoveAnchor, margin)
            cursor.insertText("# ")
            end += 2
            if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock):
                break
        cursor.endEditBlock()

    def block_uncomment(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        if cursor.position() == end: end += 1
        document = self.document()
        try:
            while cursor.position() < end:
                while document.characterAt(cursor.position()) in (' ', '\t'):
                    if not cursor.movePosition(QTextCursor.MoveOperation.NextCharacter): raise ValueError
                if document.characterAt(cursor.position()) == '#':
                    cursor.deleteChar()
                    end -= 1
                    if document.characterAt(cursor.position()) == ' ':
                        cursor.deleteChar()
                        end -= 1
                if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock): raise ValueError
        except ValueError:
            pass
        cursor.endEditBlock()

    def block_comment_toggle(self):
        incomment = False
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        if cursor.position() == end: end += 1
        document = self.document()
        try:
            while cursor.position() < end:
                while document.characterAt(cursor.position()) in (' ', '\t'):
                    if not cursor.movePosition(QTextCursor.MoveOperation.NextCharacter): raise ValueError
                if document.characterAt(cursor.position()) == '#':
                    incomment = True
                elif not cursor.atBlockEnd():
                    incomment = False
                    break
                if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock): raise ValueError
        except ValueError:
            pass
        if incomment:
            self.block_uncomment()
        else:
            self.block_comment()

    def join_lines(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        if cursor.atEnd(): return
        document = self.document()
        cursor.beginEditBlock()
        cursor.deleteChar()
        while document.characterAt(cursor.position()) in ' \t':
            cursor.deleteChar()
        cursor.insertText(' ')
        cursor.endEditBlock()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab) or \
                key == Qt.Key.Key_Backspace and modifiers != (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            cursor = self.textCursor()
            if cursor.hasSelection():
                if key == Qt.Key.Key_Tab:
                    indent(self)
                    return
                elif key == Qt.Key.Key_Backtab:
                    unindent(self)
                    return
            elif key == Qt.Key.Key_Backtab:
                unindent(self)
                return
            else:
                col = cursor.positionInBlock()
                inindent = not cursor.block().text()[:col].strip()
                if inindent:
                    if key == Qt.Key.Key_Tab:
                        indent(self, col)
                        return
                    else:
                        if not (cursor.atBlockStart()):
                            unindent(self, col)
                            return
        elif key == Qt.Key.Key_Home and not modifiers & ~Qt.KeyboardModifier.ShiftModifier:
            cursor = self.textCursor()
            txt = cursor.block().text()
            col = cursor.positionInBlock()
            mode = QTextCursor.MoveMode.KeepAnchor if modifiers & Qt.KeyboardModifier.ShiftModifier else QTextCursor.MoveMode.MoveAnchor
            if txt[:col].strip() or (col == 0 and txt.strip()):
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, mode)
                while self.document().characterAt(cursor.position()) in [' ', '\t']:
                    cursor.movePosition(QTextCursor.MoveOperation.Right, mode)
                self.setTextCursor(cursor)
                return

        super().keyPressEvent(event)

        if key in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Colon):
            autoindent(self)

    def rehighlight(self, *args, **kwargs):
        syntax = get_syntax(*args, **kwargs)
        self.highlighter = SyntaxHighlighter(self.document(), *load_syntax(syntax, PYTHON_SCHEME), default_font=EDITOR_FONT)
        self.highlighter.rehighlight()

    def reconfig(self, *args, **kwargs):
        super().reconfig()
        update_brackets_colors()
        if self.highlighter is not None:
            with BlockQtSignals(editor):
                update_python_scheme()
                self.rehighlight(*args, **kwargs)
