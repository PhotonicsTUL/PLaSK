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

import re

from ...qt import QtGui

indent_re = re.compile(r'.*<(?:[^/].*)?[^/]\s*>\s*$')
unindent_re = re.compile(r'\s*</.+\s*>.*$')


def indent(editor, col=0):
    """Indent current or selected lines unconditionally"""
    cursor = editor.textCursor()
    if cursor.hasSelection():
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
        if cursor.position() == end: end += 1
        while cursor.position() < end:
            cursor.insertText('  ')
            end += 2
            if not cursor.movePosition(QtGui.QTextCursor.NextBlock):
                break
        cursor.endEditBlock()
    else:
        cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
        cursor.insertText(' ' * (2 - (col % 2)))


def unindent(editor, col=2):
    """Unindent current or selected lines unconditionally"""
    cursor = editor.textCursor()
    cursor.beginEditBlock()
    try:
        document = editor.document()
        if cursor.hasSelection():
            start = cursor.selectionStart()
            end = cursor.selectionEnd()
            cursor.setPosition(start)
            cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
            if cursor.position() == end: end += 1
            while cursor.position() < end:
                for i in range(2):
                    if document.characterAt(cursor.position()) != ' ': break
                    cursor.deleteChar()
                    end -= 1
                if not cursor.movePosition(QtGui.QTextCursor.NextBlock): raise ValueError
        else:
            cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
            for i in range(min(col,2)):
                if document.characterAt(cursor.position()) != ' ':
                    break
                cursor.deleteChar()
    except ValueError:
        pass
    cursor.endEditBlock()


def _find_prev_indent_level(document, row, spaces):
    ni = len(spaces)
    while row > 0:
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
    curline = document.findBlockByNumber(row).text()
    spaces = curline[:len(curline) - len(curline.lstrip())]
    if cursor.atBlockStart():
        if indent_re.match(curline):
            cursor.insertText(spaces + '  ')
        elif unindent_re.match(curline):
            if row == 0:
                prevlen = 0
                pspaces = ''
            else:
                prevline = document.findBlockByNumber(row-1).text()
                pspaces = prevline[:len(prevline) - len(prevline.lstrip())]
                prevlen = len(_find_prev_indent_level(document, row-1, pspaces))
            if prevlen != len(spaces):
                cursor.insertText(spaces[:-2])
                nl = len(spaces) - len(_find_prev_indent_level(document, row+1, spaces))
                cursor.movePosition(QtGui.QTextCursor.Up)
                cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
                for i in range(nl):
                    cursor.deleteChar()
            else:
                cursor.insertText(spaces)
        else:
            cursor.insertText(spaces)
    cursor.endEditBlock()