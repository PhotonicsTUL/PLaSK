#coding utf:8

from PyQt4 import QtCore, QtGui

class OutputWindow(QtGui.QMainWindow):
    '''Main Qt window class'''

    def __init__(self, filename):
        super(OutputWindow, self).__init__()

        self.setWindowTitle(filename)

        font = QtGui.QFont()
        if sys.platform == 'win32':
            font.setFamily("Consolas")
        elif sys.platform == 'darwin':
            font.setFamily("Monaco")
        else:
            font.setFamily("Monospace")
        font.setStyleHint(QtGui.QFont.TypeWriter)
        font.setPointSize(10)
        self.messages = QtGui.QTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setAcceptRichText(True)
        self.messages.setFont(font)

        self.halt_action = QtGui.QAction(QtGui.QIcon.fromTheme(QtGui.QStyle.SP_MediaStop), "Halt", self)
        toolbar = self.addToolBar()
        toolbar.addAction(self.halt_action)

        self.setCentralWidget(self.messages)

