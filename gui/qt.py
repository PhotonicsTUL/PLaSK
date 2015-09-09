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

try:
    import matplotlib
except ImportError:
    QT_API = 'PySide'
else:
    matplotlib.use('Qt4Agg')
    QT_API = os.environ.get('QT_API')
    if QT_API is not None:
        QT_API = dict(pyqt='PyQt4', pyside='PySide', pyqt5='PyQt5').get(QT_API, 'PySide')
    else:
        QT_API = matplotlib.rcParams['backend.qt4']

if QT_API == 'PySide':
    from PySide import QtCore, QtGui
    QtSignal = QtCore.Signal
else:
    import sip
    for n in ("QString", "QVariant"):
        try:
            sip.setapi(n, 2)
        except:
            pass
    from PyQt4 import QtCore, QtGui
    QtSignal = QtCore.pyqtSignal

sys.modules['gui.qt.QtCore'] = QtCore
sys.modules['gui.qt.QtGui'] = QtGui
__all__ = ['QtCore', 'QtGui', 'QT_API', 'QtSignal']
