import sip
for n in ("QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"):
    sip.setapi(n, 2)
from PyQt4 import QtCore, QtGui

qt = 'PyQt4'

import sys
sys.modules['gui.qt.QtCore'] = QtCore
sys.modules['gui.qt.QtGui'] = QtGui
__all__ = ['QtCore', 'QtGui', 'qt']
