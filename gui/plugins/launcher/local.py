#coding utf:8

import sys

from PyQt4 import QtCore, QtGui

from gui.launch import LAUNCHERS
from gui.utils.config import CONFIG

LAUNCHERS.inser(0, ("Local process",))


class OutputWindow(QtGui.QMainWindow):
    '''Main Qt window class'''

    def __init__(self, filename, parent=None):
        super(OutputWindow, self).__init__(parent)

        self.setWindowTitle(filename)

        font = QtGui.QFont()
        font_family = CONFIG['launcher/local/font_family']
        if font_family is None:
            if sys.platform == 'win32':
                font_family = "Consolas"
            elif sys.platform == 'darwin':
                font_family = "Monaco"
            else:
                font_family = "Monospace"
            CONFIG['launcher/local/font_family'] = font_family
            font.setStyleHint(QtGui.QFont.TypeWriter)
        font.setFamily(font_family)
        font.setPointSize(CONFIG('launcher/local/font_size', 10))
        self.messages = QtGui.QTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setAcceptRichText(True)
        self.messages.setFont(font)

        self.halt_action = QtGui.QAction(QtGui.QIcon.fromTheme(QtGui.QStyle.SP_MediaStop), "Halt", self)
        toolbar = self.addToolBar()
        toolbar.addAction(self.halt_action)

        self.setCentralWidget(self.messages)

