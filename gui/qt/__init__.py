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


import sys
import os

if sys.modules.get("PyQt6.QtCore"):
    QT_API = 'pyqt6'
elif sys.modules.get("PySide6.QtCore"):
    QT_API = 'pyside6'
elif sys.modules.get("PyQt5.QtCore"):
    QT_API = 'pyqt5'
elif sys.modules.get("PySide2.QtCore"):
    QT_API = 'pyside2'
else:
    QT_API = os.environ.get('PLASK_QT_API', os.environ.get('QT_API'))

if QT_API is not None:
    QT_API = dict(pyqt6='PyQt6', pyside6='PySide6', pyqt5='PyQt5', pyside2='PySide2').get(QT_API)
    if QT_API is None:
        import warnings
        warnings.warn("Ignoring unknown PLASK_QT_API/QT_API environmental variable!")

import matplotlib

for QT_API in (QT_API, 'PyQt6', 'PySide6', 'PyQt5', 'PySide2'):
    if QT_API is None:
        continue
    elif QT_API == 'PyQt6':
        try: from PyQt6 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError:
            if 'PyQt6' in sys.modules: del sys.modules['PyQt6']
        else:
            matplotlib.use('QtAgg')
            break
    elif QT_API == 'PySide6':
        try: from PySide6 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError:
            if 'PySide6' in sys.modules: del sys.modules['PySide6']
        else:
            matplotlib.use('QtAgg')
            break
    elif QT_API == 'PyQt5':
        try: from PyQt5 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError:
            if 'PyQt5' in sys.modules: del sys.modules['PyQt5']
        else:
            try:
                matplotlib.use('Qt5Agg')
            except ValueError:
                matplotlib.use('QtAgg')
            break
    else:  # QT_API == 'PySide2':
        try: from PySide2 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError:
            if 'PySide2' in sys.modules: del sys.modules['PySide2']
        else:
            try:
                matplotlib.use('Qt5Agg')
            except ValueError:
                matplotlib.use('QtAgg')
            break

if QT_API in ('PyQt5', 'PyQt6'):
    QtSignal = QtCore.pyqtSignal
    QtSlot = QtCore.pyqtSlot
else:
    QtSignal = QtCore.Signal
    QtSlot = QtCore.Slot

if os.name == 'nt':
    QtWidgets.QApplication.addLibraryPath(os.path.join(sys.prefix, 'Library', 'plugins'))
    QtWidgets.QApplication.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), 'plugins'))


if QT_API in ('PyQt5', 'PySide2'):
    def qt_exec(self, *args, **kwargs):
        return self.exec_(*args, **kwargs)
    if not hasattr(QtGui.QFontMetrics, 'horizontalAdvance'):
        QtGui.QFontMetrics.horizontalAdvance = lambda self, *args, **kwargs: self.width(*args, **kwargs)
else:
    def qt_exec(self, *args, **kwargs):
        return self.exec(*args, **kwargs)


os.environ['QT_API'] = QT_API.lower()

sys.modules['gui.qt.QtCore'] = QtCore
sys.modules['gui.qt.QtWidgets'] = QtWidgets
sys.modules['gui.qt.QtGui'] = QtGui
sys.modules['gui.qt.QtHelp'] = QtHelp
__all__ = ['QtCore', 'QtWidgets', 'QtGui', 'QtHelp', 'QT_API', 'QtSignal', 'QtSlot', 'qt_exec']
