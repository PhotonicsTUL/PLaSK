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

try:
    qt4 = sys.qt4
except (ImportError, AttributeError):
    qt4 = 'PySide'

if qt4 == 'PySide':
    from PySide import QtCore, QtGui
else:
    import sip
    for n in ("QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"):
        sip.setapi(n, 2)
    from PyQt4 import QtCore, QtGui

sys.modules['gui.qt.QtCore'] = QtCore
sys.modules['gui.qt.QtGui'] = QtGui
__all__ = ['QtCore', 'QtGui', 'qt']
