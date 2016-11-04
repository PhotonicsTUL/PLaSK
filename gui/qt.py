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

QT_API = os.environ.get('QT_API')
if QT_API is not None:
    QT_API = dict(pyqt='PyQt4', pyqt5='PyQt5', pyside='PySide').get(QT_API, 'PySide')
    try:
        import matplotlib
    except ImportError:
        pass
    else:
        if QT_API == 'PyQt5':
            matplotlib.use('Qt5Agg')
        else:
            matplotlib.rcParams['backend.qt4'] = QT_API
            matplotlib.use('Qt4Agg')
else:
    try:
        import matplotlib
    except ImportError:
        QT_API = 'PySide'
    else:
        if matplotlib.rcParams['backend'] == 'Qt5Agg':
            QT_API = 'PyQt5'
        else:
            matplotlib.use('Qt4Agg')
            QT_API = matplotlib.rcParams['backend.qt4']

for QT_API in (QT_API, 'PySide', 'PyQt4', 'PyQt5'):
    if QT_API == 'PySide':
        try:
            from PySide import QtCore, QtGui
        except ImportError:
            pass
        else:
            QtWidgets = QtGui
            QtSignal = QtCore.Signal
            QtSlot = QtCore.Slot
            break
    elif QT_API == 'PyQt5':
        try:
            # import sip
            # for n in ("QString", "QVariant"):
            #     try:
            #         sip.setapi(n, 2)
            #     except:
            #         pass
            from PyQt5 import QtCore, QtWidgets, QtGui
        except ImportError:
            pass
        else:
            QtSignal = QtCore.pyqtSignal
            QtSlot = QtCore.pyqtSlot
            break
    else:
        try:
            import sip
            for n in ("QString", "QVariant"):
                try:
                    sip.setapi(n, 2)
                except:
                    pass
            from PyQt4 import QtCore, QtGui
        except ImportError:
            pass
        else:
            QtWidgets = QtGui
            QtSignal = QtCore.pyqtSignal
            QtSlot = QtCore.pyqtSlot
            break

sys.modules['gui.qt.QtCore'] = QtCore
sys.modules['gui.qt.QtWidgets'] = QtWidgets
sys.modules['gui.qt.QtGui'] = QtGui
__all__ = ['QtCore', 'QtWidgets', 'QtGui', 'QT_API', 'QtSignal', 'QtSlot']
