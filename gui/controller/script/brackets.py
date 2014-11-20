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

# Bracket highliter. Calculates list of QTextEdit.ExtraSelection.
# Based on https://github.com/hlamer/qutepart

from ...qt import QtGui
from ...utils.config import CONFIG


MATCHING_BRACKET_COLOR = QtGui.QColor(CONFIG('editor/matching_bracket_color', '#aaffaa'))
NOT_MATCHING_BRACKET_COLOR = QtGui.QColor(CONFIG('editor/not_matching_bracket_color', '#ffaaaa'))


_START_BRACKETS = '({['
_END_BRACKETS=')}]'
_ALL_BRACKETS = _START_BRACKETS + _END_BRACKETS
_OPOSITE_BRACKET = dict(
    (br, op) for (br, op) in zip(_START_BRACKETS + _END_BRACKETS, _END_BRACKETS + _START_BRACKETS))


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


def _find_matching(bracket, block, column):
    """Find matching bracket for the bracket."""
    if bracket in _START_BRACKETS:
        chars = _iterate_forward(block, column + 1)
    else:
        chars = _iterate_backward(block, column)

    depth = 1
    ignore = False
    oposite = _OPOSITE_BRACKET[bracket]
    for block, column, char in chars:
        if char in ("'", '"'):
            if ignore:
                if ignore == char:
                    ignore = False
            else:
                ignore = char
        if not ignore:
            if char == oposite:
                depth -= 1
                if depth == 0:
                    return block, column
            elif char == bracket:
                depth += 1
    else:
        return None, None


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


def _highlight_bracket(bracket, block, column):
    """Highlight bracket and matching bracket. Return tuple of QTextEdit.ExtraSelection's"""
    matched_block, matched_column = _find_matching(bracket, block, column)

    if matched_block is not None:
        return [_make_match_selection(block, column, True),
                _make_match_selection(matched_block, matched_column, True)]
    else:
        return [_make_match_selection(block, column, False)]


def get_selections(editor, block, column):
    """List of QTextEdit.ExtraSelection's, which highlights brackets"""
    text = block.text()

    if column > 0 and text[column - 1] in _ALL_BRACKETS:
        return _highlight_bracket(text[column - 1], block, column - 1)
    elif column < len(text) and text[column] in _ALL_BRACKETS:
        return _highlight_bracket(text[column], block, column)
    else:
        return []
