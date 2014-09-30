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

from ...utils.textedit import TextEdit
from ...qt import QtCore, QtGui

from ...utils.config import CONFIG

MATCHING_BRACKET_COLOR = QtGui.QColor(CONFIG('editor/matching_bracket_color', '#aaffaa'))
NOT_MATCHING_BRACKET_COLOR = QtGui.QColor(CONFIG('editor/not_matching_bracket_color', '#ffaaaa'))


class ScriptEditor(TextEdit):
    """Editor with some features usefult for script editing"""

    def __init__(self, controller, parent=None):
        self.controller = controller
        super(ScriptEditor, self).__init__(parent)

        self.cursorPositionChanged.connect(self.update_selections)

        comment_action = QtGui.QAction('Comm&ent lines', self)
        uncomment_action = QtGui.QAction('Uncomm&ent lines', self)
        comment_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Slash)
        uncomment_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Slash)
        comment_action.triggered.connect(self.comment_block)
        uncomment_action.triggered.connect(self.uncomment_block)
        # self.context_menu.addSeparator()
        self.addAction(comment_action)
        self.addAction(uncomment_action)


    def update_selections(self):
        """Add our own custom selections"""
        cursor_column = self.textCursor().positionInBlock()
        brackets = BracketHighlighter.get_selections(self, self.textCursor().block(), cursor_column)
        self.setExtraSelections(self.extraSelections() + brackets)

    def comment_block(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
        while cursor.position() < end:
            cursor.insertText("# ")
            cursor.movePosition(QtGui.QTextCursor.NextBlock)
        cursor.endEditBlock()

    def uncomment_block(self):
        # cursor = self.textCursor()
        # cursor.beginEditBlock()
        # start = cursor.selectionStart()
        # end = cursor.selectionEnd()
        # cursor.setPosition(start)
        # cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
        # while cursor.position() < end:
        #     cursor.insertText("# ")
        #     cursor.movePosition(QtGui.QTextCursor.NextBlock)
        # cursor.endEditBlock()
        pass

class BracketHighlighter(object):
    """ Bracket highliter.
        Calculates list of QTextEdit.ExtraSelection.
        Based on https://github.com/hlamer/qutepart
    """
    _START_BRACKETS = '({['
    _END_BRACKETS = ')}]'
    _ALL_BRACKETS = _START_BRACKETS + _END_BRACKETS
    _OPOSITE_BRACKET = dict(
        (br, op) for (br, op) in zip(_START_BRACKETS + _END_BRACKETS, _END_BRACKETS + _START_BRACKETS))

    @staticmethod
    def _iterate_forward(block, start):
        """Traverse document forward. Yield (block, column, char)"""
        # Chars in the start line
        for column, char in list(enumerate(block.text()))[start:]:
            yield block, column, char
        block = block.next()

        # Next lines
        while block.isValid():
            for column, char in enumerate(block.text()):
                yield block, column, char

            block = block.next()

    @staticmethod
    def _iterate_backward(block, start):
        """Traverse document forward. Yield (block, column, char)"""
        # Chars in the start line
        for column, char in reversed(list(enumerate(block.text()[:start]))):
            yield block, column, char
        block = block.previous()

        # Next lines
        while block.isValid():
            for column, char in reversed(list(enumerate(block.text()))):
                yield block, column, char

            block = block.previous()

    @staticmethod
    def _find_matching(bracket, block, column):
        """Find matching bracket for the bracket."""
        if bracket in BracketHighlighter._START_BRACKETS:
            chars = BracketHighlighter._iterate_forward(block, column + 1)
        else:
            chars = BracketHighlighter._iterate_backward(block, column)

        depth = 1
        ignore = False
        oposite = BracketHighlighter._OPOSITE_BRACKET[bracket]
        for block, column, char in chars:
            if char in ("'", '"'):
                ignore = not ignore
            if not ignore:
                if char == oposite:
                    depth -= 1
                    if depth == 0:
                        return block, column
                elif char == bracket:
                    depth += 1
        else:
            return None, None

    @staticmethod
    def _make_match_selection(block, column, matched):
        """Make matched or unmatched QTextEdit.ExtraSelection
        """
        selection = QtGui.QTextEdit.ExtraSelection()

        if matched:
            bg_color = MATCHING_BRACKET_COLOR
        else:
            bg_color = NOT_MATCHING_BRACKET_COLOR

        selection.format.setBackground(bg_color)
        selection.cursor = QtGui.QTextCursor(block)
        selection.cursor.setPosition(block.position() + column)
        selection.cursor.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.KeepAnchor)

        return selection

    @staticmethod
    def _highlight_bracket(bracket, block, column):
        """Highlight bracket and matching bracket. Return tuple of QTextEdit.ExtraSelection's"""
        matched_block, matched_column = BracketHighlighter._find_matching(bracket, block, column)

        if matched_block is not None:
            return [BracketHighlighter._make_match_selection(block, column, True),
                    BracketHighlighter._make_match_selection(matched_block, matched_column, True)]
        else:
            return [BracketHighlighter._make_match_selection(block, column, False)]

    @staticmethod
    def get_selections(editor, block, column):
        """List of QTextEdit.ExtraSelection's, which highlights brackets"""
        text = block.text()

        if column > 0 and text[column - 1] in BracketHighlighter._ALL_BRACKETS:
            return BracketHighlighter._highlight_bracket(text[column - 1], block, column - 1)
        elif column < len(text) and text[column] in BracketHighlighter._ALL_BRACKETS:
            return BracketHighlighter._highlight_bracket(text[column], block, column)
        else:
            return []
