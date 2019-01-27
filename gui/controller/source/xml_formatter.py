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
from xml.parsers import expat

from ...qt.QtGui import *

indent_re = re.compile(r'.*<[^/><]+>\s*$')


def indent(editor, col=0):
    """Indent current or selected lines unconditionally"""
    cursor = editor.textCursor()
    if cursor.hasSelection():
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.StartOfBlock)
        if cursor.position() == end: end += 1
        while cursor.position() < end:
            cursor.insertText('  ')
            end += 2
            if not cursor.movePosition(QTextCursor.NextBlock):
                break
        cursor.endEditBlock()
    else:
        cursor.movePosition(QTextCursor.StartOfBlock)
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
            cursor.movePosition(QTextCursor.StartOfBlock)
            if cursor.position() == end: end += 1
            while cursor.position() < end:
                for i in range(2):
                    if document.characterAt(cursor.position()) != ' ': break
                    cursor.deleteChar()
                    end -= 1
                if not cursor.movePosition(QTextCursor.NextBlock): raise ValueError
        else:
            cursor.movePosition(QTextCursor.StartOfBlock)
            for i in range(min(col,2)):
                if document.characterAt(cursor.position()) != ' ':
                    break
                cursor.deleteChar()
    except ValueError:
        pass
    cursor.endEditBlock()


def indent_new_line(editor):
    """Automatically set indentation of the current line"""
    cursor = editor.textCursor()
    row = cursor.blockNumber() - 1
    if row == -1:
        return
    cursor.joinPreviousEditBlock()
    document = editor.document()
    curline = document.findBlockByNumber(row).text()
    spaces = curline[:len(curline) - len(curline.lstrip())]
    if indent_re.match(curline):
        cursor.insertText(spaces + '  ')
    else:
        cursor.insertText(spaces)
    cursor.endEditBlock()


def parse_slash(editor):
    """Close the current tag, and unindent closing tag if necessary"""
    cursor = editor.textCursor()
    pos = cursor.position()
    text = editor.toPlainText()[:pos]
    closing = text[-1] == '<'
    if closing:
        text = text[:-1]
        pos -= 1
    else:
        text = text+'/>'
    parser = expat.ParserCreate('utf8')
    stack = [(None, 0, None)]
    parser.StartElementHandler = lambda tag, atr:\
        stack.append((
            tag,
            parser.CurrentColumnNumber if parser.CurrentLineNumber != stack[-1][2] else stack[-1][1],
            parser.CurrentLineNumber))
    parser.EndElementHandler = lambda tag: stack.pop()
    try:
        parser.Parse('<text>\n'+text)
    except expat.ExpatError:
        return False
    if closing:
        if len(stack) < 2:
            return False
        tag, col, _ = stack[-1]
        cursor.beginEditBlock()
        cursor.insertText('/'+tag+'>')
        cur = cursor.block().position() + col
        if cur < pos and text[cur:].strip() == '':
            cursor.setPosition(cur)
            cursor.setPosition(pos, QTextCursor.KeepAnchor)
            if cursor.hasSelection():
                cursor.deleteChar()
        cursor.endEditBlock()
        return True
    elif parser.CurrentLineNumber == cursor.blockNumber() + 2 and \
         parser.CurrentColumnNumber == pos - cursor.block().position() + 2:
        cursor.joinPreviousEditBlock()
        cursor.insertText('/>')
        cursor.endEditBlock()
        return True
    return False
