import sip
for n in ("QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"):
    sip.setapi(n, 2)
from PyQt4 import Qt, QtCore, QtGui

# from PySide import QtCore, QtGui
# from PySide.QtCore import Qt

import sys
sys.modules['gui.qt.Qt'] = Qt
sys.modules['gui.qt.QtCore'] = QtCore
sys.modules['gui.qt.QtGui'] = QtGui
__all__ = ['Qt', 'QtCore', 'QtGui']
