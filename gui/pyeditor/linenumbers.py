import math

from ..qt import Qt, QtCore, QtGui




class LineNumberArea(QtGui.QWidget):
    """Line numbers widget

       http://qt-project.org/doc/qt-4.8/widgets-codeeditor.html
    """

    def __init__(self, code_editor):
        self._code_editor = code_editor

    def get_width(self):
        """Return required width"""
        count = max(1, self._code_editor.blockCount())
        digits = int(math.log10(count)) + 1
        return 3 + self._code_editor.fontMetrics().width('9') * digits

    def sizeHint(self):
        QtCore.QSize(self.get_width(), 0)

    def update_width(self):
        self._code_editor.setViewportMargins(self.get_width(), 0, 0, 0)

    def update_area(self, rect, dy):
        if dy:
            self.scroll(0, dy)
        else:
            self.update(0, rect.y(), self.width(), rect.height())

        if rect.contains(self._code_editor.viewport().rect()):
            self.update_width()

    def paintEvent(self, event):
        self._code_editor.lineNumberAreaPaintEvent(event)
