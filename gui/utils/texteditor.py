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

import math

from ..qt import QtCore, QtGui
from ..qt.QtCore import Qt

from .widgets import EDITOR_FONT
from .config import CONFIG


def update_textedit_colors():
    global CURRENT_LINE_COLOR, SELECTION_COLOR
    CURRENT_LINE_COLOR = QtGui.QColor(CONFIG['editor/current_line_color'])
    SELECTION_COLOR = QtGui.QColor(CONFIG['editor/selection_color'])
update_textedit_colors()


class TextEditor(QtGui.QPlainTextEdit):
    """Improved editor with line numbers and some other neat stuff"""

    def __init__(self, parent=None, line_numbers=True):
        super(TextEditor, self).__init__(parent)
        self.setFont(EDITOR_FONT)
        self.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        if line_numbers:
            self.line_numbers = LineNumberArea(self)
            self.line_numbers.update_width()
            self.blockCountChanged.connect(self.line_numbers.update_width)
            self.updateRequest.connect(self.line_numbers.on_update_request)
        else:
            self.line_numbers = None
        self.cursorPositionChanged.connect(self.update_selections)
        self.selectionChanged.connect(self.update_selections)
        self.selections = []

        self._changed_pos = 0
        self.textChanged.connect(self.on_text_change)

    def resizeEvent(self, e):
        super(TextEditor, self).resizeEvent(e)
        if self.line_numbers is not None:
            cr = self.contentsRect()
            self.line_numbers.setGeometry(QtCore.QRect(cr.left(), cr.top(),
                                                       self.line_numbers.get_width(), cr.height()))

    def on_text_change(self):
        self._changed_pos = self.textCursor().position()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        if key == Qt.Key_Backspace and modifiers == (Qt.ControlModifier | Qt.ShiftModifier):
            cursor = self.textCursor()
            cursor.setPosition(self._changed_pos)
            self.setTextCursor(cursor)
            event.ignore()
            return
        super(TextEditor, self).keyPressEvent(event)

    def focusInEvent(self, event):
        super(TextEditor, self).focusInEvent(event)
        self.update_selections()

    def focusOutEvent(self, event):
        super(TextEditor, self).focusOutEvent(event)
        self.update_selections()

    def update_selections(self, selections=None):
        """Add our own custom selections"""
        if selections is not None:
            self.selections = selections
        self.setExtraSelections(self.highlight_current_line() + self.get_same_as_selected() + self.selections)

    def highlight_current_line(self):
        selection = QtGui.QTextEdit.ExtraSelection()
        if self.hasFocus():
            selection.format.setBackground(CURRENT_LINE_COLOR)
        selection.format.setProperty(QtGui.QTextFormat.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        return [selection]

    def get_same_as_selected(self):
        cursor = self.textCursor()
        if not cursor.hasSelection(): return []
        document = self.document()
        text = cursor.selectedText()
        if not text.strip(): return []
        cursor.movePosition(QtGui.QTextCursor.Start)
        selections = []
        while True:
            cursor = document.find(text, cursor,
                                   QtGui.QTextDocument.FindCaseSensitively | QtGui.QTextDocument.FindWholeWords)
            if not cursor.isNull():
                selection = QtGui.QTextEdit.ExtraSelection()
                selection.cursor = cursor
                selection.format.setBackground(SELECTION_COLOR)
                selections.append(selection)
            else:
                break
        return selections


class TextEditorWithCB(TextEditor):
    """
        TextEditor which emits some extra callbacks:
        focus_out_cb - when it lost focus
        key_cb - when kay is pressed
    """
    def __init__(self, focus_out_cb = None, key_cb = None, **kwargs):
        super(TextEditorWithCB, self).__init__(**kwargs)
        self.focus_out_cb = focus_out_cb
        self.key_cb = key_cb

    def focusOutEvent(self, event):
        super(TextEditorWithCB, self).focusOutEvent(event)
        if self.focus_out_cb is not None: self.focus_out_cb()

    def keyPressEvent(self, event):
        super(TextEditorWithCB, self).keyPressEvent(event)
        if self.key_cb is not None: self.key_cb(event)


class LineNumberArea(QtGui.QWidget):
    """Line numbers widget

       http://qt4-project.org/doc/qt4-4.8/widgets-codeeditor.html
    """

    def __init__(self, editor):
        super(LineNumberArea, self).__init__(editor)
        self.editor = editor
        self._offset = 0
        self._count_cache = -1, -1

    def get_width(self):
        """Return required width"""
        count = max(1, self.editor.blockCount() + self._offset)
        digits = int(math.log10(count)) + 1
        return 8 + self.editor.fontMetrics().width('9') * digits

    def sizeHint(self):
        QtCore.QSize(self.get_width(), 0)

    def update_width(self, n=0):
        self.editor.setViewportMargins(self.get_width(), 0, 0, 0)

    def on_update_request(self, rect, dy):
        if dy:
            self.scroll(0, dy)
        elif self._count_cache[0] != self.editor.blockCount() or\
             self._count_cache[1] != self.editor.textCursor().block().lineCount():
            self.update(0, rect.y(), self.width(), rect.height())
            self._count_cache = self.editor.blockCount(), self.editor.textCursor().block().lineCount()
        if rect.contains(self.editor.viewport().rect()):
            self.update_width()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(event.rect(), QtGui.QColor('#ddd'))
        block = self.editor.firstVisibleBlock()
        block_number = block.blockNumber() + 1 + self._offset
        top = self.editor.blockBoundingGeometry(block).translated(self.editor.contentOffset()).top()
        bottom = top + self.editor.blockBoundingRect(block).height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(QtCore.Qt.darkGray)
                painter.drawText(0, top, self.width()-3, self.editor.fontMetrics().height(),
                                 QtCore.Qt.AlignRight, str(block_number))
            block = block.next()
            top = bottom
            bottom = top + self.editor.blockBoundingRect(block).height()
            block_number += 1

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, val):
        if val is not None:
            self._offset = val
            self.update_width()