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

# coding utf:8

import sys

from gui.qt import QtCore, QtGui

from gui.launch import LAUNCHERS
from gui.utils.config import CONFIG


class Launcher(object):
    name = 'Remote Batch Job'

    def widget(self):
        widget = QtGui.QWidget()
        return widget


# LAUNCHERS.append(Launcher())
