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

import sys
import os


def _setup_matplotlib(backend):
    try:
        import matplotlib
    except ImportError:
        pass
    else:
        try:
            matplotlib.use('QtAgg')
        except ValueError:
            if backend not in ('PyQt5', 'PySide2'):
                return False
            matplotlib.use('Qt5Agg')
            try:
                matplotlib.rcParams['backend.qt5'] = backend
            except (KeyError, ValueError):
                pass
    return True


QT_API = os.environ.get('PLASK_QT_API', os.environ.get('QT_API'))

if QT_API is not None:
    QT_API = dict(pyqt5='PyQt5', pyside2='PySide2', pyqt6='PyQt6', pyside6='PySide6').get(QT_API)
    if QT_API is None:
        import warnings
        warnings.warn("Ignoring unknown QT_API environmental variable!")

for QT_API in (QT_API, 'PySide2', 'PyQt5', 'PySide6', 'PyQt6'):
    if QT_API is None or not _setup_matplotlib(QT_API):
        continue
    elif QT_API == 'PyQt6':
        try: from PyQt6 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError: pass
        else: break
    elif QT_API == 'PySide6':
        try: from PySide6 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError: pass
        else: break
    elif QT_API == 'PyQt5':
        try: from PyQt5 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError: pass
        else: break
    else:  # QT_API == 'PySide2':
        try: from PySide2 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError: pass
        else: break

if QT_API in ('PyQt5', 'PyQt6'):
    QtSignal = QtCore.pyqtSignal
    QtSlot = QtCore.pyqtSlot
else:
    QtSignal = QtCore.Signal
    QtSlot = QtCore.Slot

if os.name == 'nt' and QT_API in ('PyQt5', 'PySide2'):
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
