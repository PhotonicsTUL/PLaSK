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

QT_API = os.environ.get('PLASK_QT_API', os.environ.get('QT_API'))
if QT_API is not None:
    QT_API = dict(pyqt='PyQt4v2', pyqt4='PyQt4v2', pyqt4v2='PyQt4v2', pyside='PySide',
                  pyqt5='PyQt5', pyside2='PySide2').get(QT_API)
    if QT_API is None:
        import warnings
        warnings.warn("Ignoring unknown QT_API environmental variable!")
    else:
        try: _mplbackend = os.environ.pop('MPLBACKEND')
        except KeyError: _mplbackend = None
        try:
            import matplotlib
        except ImportError:
            pass
        else:
            if QT_API in ('PyQt5', 'PySide2'):
                matplotlib.rcParams['backend.qt5'] = QT_API
                matplotlib.use('Qt5Agg')
            else:
                matplotlib.rcParams['backend.qt4'] = QT_API
                matplotlib.use('Qt4Agg')
        finally:
            if _mplbackend is not None:
                os.environ['MPLBACKEND'] = _mplbackend
if QT_API is None:
    try:
        import matplotlib
    except ImportError:
        QT_API = 'PyQt5'
    else:
        if matplotlib.rcParams['backend'] == 'Qt4Agg':
            QT_API = matplotlib.rcParams['backend.qt4']
        else:
            matplotlib.use('Qt5Agg')
            QT_API = matplotlib.rcParams['backend.qt5']

for QT_API in (QT_API, 'PySide2', 'PyQt5', 'PySide', 'PyQt4'):
    if QT_API == 'PySide':
        try:
            from PySide import QtCore, QtGui, QtGui as QtWidgets, QtHelp
        except ImportError:
            pass
        else:
            QtSignal = QtCore.Signal
            QtSlot = QtCore.Slot
            break
    elif QT_API in ('PyQt4', 'PyQt4v2'):
        try:
            import sip
            for n in ("QString", "QVariant"):
                try:
                    sip.setapi(n, 2)
                except:
                    pass
            from PyQt4 import QtCore, QtGui, QtGui as QtWidgets, QtHelp
        except ImportError:
            pass
        else:
            QtSignal = QtCore.pyqtSignal
            QtSlot = QtCore.pyqtSlot
            break
    elif QT_API == 'PyQt5':
        try:
            from PyQt5 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError:
            pass
        else:
            if os.name == 'nt':
                QtWidgets.QApplication.addLibraryPath(os.path.join(sys.prefix, 'Library', 'plugins'))
                QtWidgets.QApplication.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), 'plugins'))
            QtSignal = QtCore.pyqtSignal
            QtSlot = QtCore.pyqtSlot
            break
    else:  # QT_API == 'PySide2':
        try:
            from PySide2 import QtCore, QtWidgets, QtGui, QtHelp
        except ImportError:
            pass
        else:
            QT_API = 'PySide2'
            if os.name == 'nt':
                QtWidgets.QApplication.addLibraryPath(os.path.join(sys.prefix, 'Library', 'plugins'))
                QtWidgets.QApplication.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), 'plugins'))
            QtSignal = QtCore.Signal
            QtSlot = QtCore.Slot
            break


sys.modules['gui.qt.QtCore'] = QtCore
sys.modules['gui.qt.QtWidgets'] = QtWidgets
sys.modules['gui.qt.QtGui'] = QtGui
sys.modules['gui.qt.QtHelp'] = QtHelp
__all__ = ['QtCore', 'QtWidgets', 'QtGui', 'QtHelp', 'QT_API', 'QtSignal', 'QtSlot']
