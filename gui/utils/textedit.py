import math

from ..qt import QtCore, QtGui
from ..utils.gui import DEFAULT_FONT


class TextEdit(QtGui.QPlainTextEdit):
    """Improved editor with line numbers and some other neat stuff"""

    def __init__(self, parent=None):
        super(TextEdit, self).__init__(parent)
        self.setFont(DEFAULT_FONT)
        self.line_numbers = LineNumberArea(self)
        self.line_numbers.update_width()
        self.blockCountChanged.connect(self.line_numbers.update_width)
        self.updateRequest.connect(self.line_numbers.on_update_request)
        #self.cursorPositionChanged.connect(self.highlight_current_line)

    def resizeEvent(self, e):
        super(TextEdit, self).resizeEvent(e)
        cr = self.contentsRect()
        self.line_numbers.setGeometry(QtCore.QRect(cr.left(), cr.top(),
                                                   self.line_numbers.get_width(), cr.height()))


class LineNumberArea(QtGui.QWidget):
    """Line numbers widget

       http://qt-project.org/doc/qt-4.8/widgets-codeeditor.html
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