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


LAUNCHERS.append(Launcher())
