from PySide import QtCore, QtGui

qt = 'PySide'

import sys
sys.modules['gui.qt.QtCore'] = QtCore
sys.modules['gui.qt.QtGui'] = QtGui
__all__ = ['QtCore', 'QtGui', 'qt']
