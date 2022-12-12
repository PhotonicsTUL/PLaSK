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


from .. import TextEditor
from ....qt.QtCore import *
from ....qt.QtWidgets import *
from ....qt.QtGui import *
from ....utils.config import CONFIG, parse_highlight, dark_style
from ....utils.qsignals import BlockQtSignals
from ....utils.widgets import EDITOR_FONT, set_icon_size
from ....lib.highlighter import SyntaxHighlighter, load_syntax
from ....lib.highlighter.xml import SYNTAX
from .formatter import indent, unindent, indent_new_line, parse_slash

XML_SCHEME = {}

def update_xml_scheme():
    global XML_SCHEME
    XML_SCHEME['syntax_comment'] = parse_highlight(CONFIG['syntax/xml_comment'])
    XML_SCHEME['syntax_tag'] = parse_highlight(CONFIG['syntax/xml_tag'])
    XML_SCHEME['syntax_attr'] = parse_highlight(CONFIG['syntax/xml_attr'])
    XML_SCHEME['syntax_value'] = parse_highlight(CONFIG['syntax/xml_value'])
    XML_SCHEME['syntax_text'] = parse_highlight(CONFIG['syntax/xml_text'])
    XML_SCHEME['syntax_define'] = parse_highlight(CONFIG['syntax/xml_define'])
update_xml_scheme()


class XMLEditor(TextEditor):

    def __init__(self, parent=None, line_numbers=True):
        super().__init__(parent, line_numbers)
        palette = self.palette()
        color = parse_highlight(CONFIG['syntax/xml_text']).get('color')
        if color is None: color = CONFIG['editor/foreground_color']
        palette.setColor(QPalette.ColorRole.Text, QColor(color))
        self.setPalette(palette)
        self.highlighter = SyntaxHighlighter(self.document(),
                                             *load_syntax(SYNTAX, XML_SCHEME),
                                             default_font=EDITOR_FONT)

    def reconfig(self):
        super().reconfig()
        if self.highlighter is not None:
            with BlockQtSignals(self):
                update_xml_scheme()
                self.highlighter = SyntaxHighlighter(self.document(),
                                                     *load_syntax(SYNTAX, XML_SCHEME),
                                                     default_font=EDITOR_FONT)

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab, Qt.Key.Key_Backspace):
            cursor = self.textCursor()
            if cursor.hasSelection():
                if key == Qt.Key.Key_Tab:
                    indent(self)
                    event.ignore()
                    return
                elif key == Qt.Key.Key_Backtab:
                    unindent(self)
                    event.ignore()
                    return
            elif key == Qt.Key.Key_Backtab:
                unindent(self)
                event.ignore()
                return
            else:
                col = cursor.positionInBlock()
                inindent = not cursor.block().text()[:col].strip()
                if inindent:
                    if key == Qt.Key.Key_Tab:
                        indent(self, col)
                        event.ignore()
                        return
                    else:
                        if not (cursor.atBlockStart()):
                            unindent(self, col)
                            event.ignore()
                            return
        elif key == Qt.Key.Key_Home and not modifiers & ~Qt.KeyboardModifier.ShiftModifier:
            cursor = self.textCursor()
            txt = cursor.block().text()
            col = cursor.positionInBlock()
            mode = QTextCursor.MoveMode.KeepAnchor if modifiers & Qt.KeyboardModifier.ShiftModifier else QTextCursor.MoveMode.MoveAnchor
            if txt[:col].strip():
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, mode)
                while self.document().characterAt(cursor.position()) in [' ', '\t']:
                    cursor.movePosition(QTextCursor.MoveOperation.Right, mode)
                self.setTextCursor(cursor)
                event.ignore()
                return
        elif key == Qt.Key.Key_Slash:
            if parse_slash(self):
                event.ignore()
                return
        elif key == Qt.Key.Key_Less:
            cursor = self.textCursor()
            if cursor.hasSelection():
                pos, anchor = cursor.position() + 1, cursor.anchor() + 1
                text = '<' + cursor.selectedText() + '>'
                cursor.insertText(text)
                cursor.setPosition(anchor)
                cursor.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)
                self.setTextCursor(cursor)
                event.ignore()
                return

        super().keyPressEvent(event)

        if key in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            indent_new_line(self)

    def reconfig(self):
        self.setFont(EDITOR_FONT)
        color = parse_highlight(CONFIG['syntax/xml_text']).get('color')
        if color is None: color = CONFIG['editor/foreground_color']
        self.setStyleSheet("QPlainTextEdit {{ color: {fg}; background-color: {bg} }}".format(
            fg=color, bg=CONFIG['editor/background_color']
        ))
        self.line_numbers.setFont(EDITOR_FONT)
