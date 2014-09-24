import math

from ..qt import Qt, QtCore, QtGui
from ..utils.gui import DEFAULT_FONT


class TextEdit(QtGui.QPlainTextEdit):
    """Improved editor with line numebers and some other neat stuff"""

    def __init__(self, parent=None):
        super(TextEdit, self).__init__(parent)
        self.setFont(DEFAULT_FONT)

        self.line_numbers = LineNumberArea(self)

# connect(this, SIGNAL(blockCountChanged(int)), this, SLOT(updateLineNumberAreaWidth(int)));
#      connect(this, SIGNAL(updateRequest(QRect,int)), this, SLOT(updateLineNumberArea(QRect,int)));
#      connect(this, SIGNAL(cursorPositionChanged()), this, SLOT(highlightCurrentLine()));
#
#      updateLineNumberAreaWidth(0);

# void CodeEditor::resizeEvent(QResizeEvent *e)
#  {
#      QPlainTextEdit::resizeEvent(e);
#
#      QRect cr = contentsRect();
#      lineNumberArea->setGeometry(QRect(cr.left(), cr.top(), lineNumberAreaWidth(), cr.height()));
#  }


class LineNumberArea(QtGui.QWidget):
    """Line numbers widget

       http://qt-project.org/doc/qt-4.8/widgets-codeeditor.html
    """

    def __init__(self, editor):
        super(LineNumberArea, self).__init__(editor)
        self.editor = editor


    def get_width(self):
        """Return required width"""
        count = max(1, self.editor.blockCount())
        digits = int(math.log10(count)) + 1
        return 3 + self.editor.fontMetrics().width('9') * digits

    def sizeHint(self):
        QtCore.QSize(self.get_width(), 0)

    def update_width(self):
        self.editor.setViewportMargins(self.get_width(), 0, 0, 0)

    def update_area(self, rect, dy):
        if dy:
            self.scroll(0, dy)
        else:
            self.update(0, rect.y(), self.width(), rect.height())

        if rect.contains(self.editor.viewport().rect()):
            self.update_width()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(event.rect(), QtGui.QColor('lightgray'))
        block = self.editor.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.editor.blockBoundingGeometry(block).translated(self.editor.contentOffset()).top()
        bottom = top + self.editor.blockBoundingRect(block).height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QtGui.QColor('black'))
                painter.drawText(0, top, self.width(), self.editor.fontMetrics().height(), Qt.AlignRight, number)
            block = block.next()
            top = bottom
            bottom = top + self.editor.blockBoundingRect(block).height()
            block_number += 1