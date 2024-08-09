# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


import re

from ....qt.QtGui import *

indent_re = re.compile(r'''([^#'"]|"[^"]"|'[^']')+:\s*(#.*)?$''')
prev_unindent_re = re.compile(r'''([^#'"]|"[^"]"|'[^']')*'''
                              r'''((return|raise)(\s([^#'"]|"[^"]"|'[^']')+)?|pass|break|continue)\s*(#.*)?$''')
current_unindent_re = re.compile(r'''([^#'"]|"[^"]"|'[^']')*'''
                                 r'''((elif|except)(\s([^#'"]|"[^"]"|'[^']')+)?|else|finally):\s*(#.*)?$''')


def indent(editor, col=0):
    """Indent current or selected lines unconditionally"""
    cursor = editor.textCursor()
    if cursor.hasSelection():
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        if cursor.position() == end: end += 1
        while cursor.position() < end:
            cursor.insertText('    ')
            end += 4
            if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock):
                break
        cursor.endEditBlock()
    else:
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        cursor.insertText(' ' * (4 - (col % 4)))


def unindent(editor, col=4):
    """Unindent current or selected lines unconditionally"""
    cursor = editor.textCursor()
    cursor.beginEditBlock()
    try:
        document = editor.document()
        if cursor.hasSelection():
            start = cursor.selectionStart()
            end = cursor.selectionEnd()
            cursor.setPosition(start)
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            if cursor.position() == end: end += 1
            while cursor.position() < end:
                for i in range(4):
                    if document.characterAt(cursor.position()) != ' ': break
                    cursor.deleteChar()
                    end -= 1
                if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock): raise ValueError
        else:
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
            for i in range(min(col,4)):
                if document.characterAt(cursor.position()) != ' ':
                    break
                cursor.deleteChar()
    except ValueError:
        pass
    cursor.endEditBlock()


def _find_prev_indent_level(document, row, spaces):
    ni = len(spaces)
    while row != 0:
        row -= 1
        line = document.findBlockByNumber(row).text()
        nl = len(line) - len(line.lstrip())
        if nl < ni:
            return spaces[:nl]
    return spaces


def autoindent(editor):
    """Automatically set indentation of the current line"""
    cursor = editor.textCursor()
    document = editor.document()
    row = cursor.blockNumber() - 1
    if row == -1:
        return
    cursor.joinPreviousEditBlock()
    prevline = document.findBlockByNumber(row).text()
    spaces = prevline[:len(prevline) - len(prevline.lstrip())]
    if cursor.atBlockStart():
        if indent_re.match(prevline):
            cursor.insertText(spaces + '    ')
        elif prev_unindent_re.match(prevline):
            spaces = _find_prev_indent_level(document, row, spaces)
            cursor.insertText(spaces)
        else:
            cursor.insertText(spaces)
    else:
        curline = document.findBlockByNumber(row+1).text()
        if current_unindent_re.match(curline):
            cspaces = curline[:len(curline) - len(curline.lstrip())]
            prevlen = len(_find_prev_indent_level(document, row, spaces))
            if indent_re.match(prevline):
                prevlen += 4
            if prevlen != len(cspaces):
                nl = len(cspaces) - len(_find_prev_indent_level(document, row+1, cspaces))
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
                for i in range(nl):
                    cursor.deleteChar()
    cursor.endEditBlock()
