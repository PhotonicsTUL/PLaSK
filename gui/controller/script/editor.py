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

from ...qt import QtGui
from ...qt.QtCore import Qt
from ...utils.textedit import TextEdit

from .brackets import get_selections as get_bracket_selections
from .indenter import indent, unindent, autoindent


class ScriptEditor(TextEdit):
    """Editor with some features usefult for script editing"""

    def __init__(self, parent=None, controller=None):
        self.controller = controller
        super(ScriptEditor, self).__init__(parent)

        self.cursorPositionChanged.connect(self.update_selections)

        self.comment_action = QtGui.QAction('Co&mment lines', self)
        self.uncomment_action = QtGui.QAction('Uncomm&ent lines', self)
        self.comment_action.setShortcut(Qt.CTRL + Qt.Key_Slash)
        self.uncomment_action.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_Slash)
        self.comment_action.triggered.connect(self.block_comment)
        self.uncomment_action.triggered.connect(self.block_uncomment)
        self.addAction(self.comment_action)
        self.addAction(self.uncomment_action)

    def keyPressEvent(self, event):
        key = event.key()
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
        super(ScriptEditor, self).keyPressEvent(event)
        if key in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Colon):
            autoindent(self)

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
