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

import shlex

from .qt import QtGui


LAUNCHERS = []
_launch_args = ''


class LaunchDialog(QtGui.QDialog):

    def __init__(self, window, parent=None):
        super(LaunchDialog, self).__init__(parent)
        self.setWindowTitle("Launch Computations")

        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        combo = QtGui.QComboBox()
        combo.insertItems(len(LAUNCHERS), [item.name for item in LAUNCHERS])
        combo.currentIndexChanged.connect(self.launcher_changed)
        self.layout.addWidget(combo)

        self.layout.addWidget(QtGui.QLabel("Command line argument:"))
        self.args = QtGui.QLineEdit()
        self.args.setText(_launch_args)
        self.layout.addWidget(self.args)

        self.launcher_widgets = [l.widget(window) for l in LAUNCHERS]
        for widget in self.launcher_widgets:
            widget.setVisible(False)
            self.layout.addWidget(widget)
        self.current = combo.currentIndex()
        self.launcher_widgets[self.current].setVisible(True)

        self.setMinimumWidth(4*QtGui.QFontMetrics(QtGui.QFont()).width(self.windowTitle()))

        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui. QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def launcher_changed(self, index):
        self.launcher_widgets[self.current].setVisible(False)
        self.current = index
        self.launcher_widgets[self.current].setVisible(True)
        self.resize(self.width(), 0)


def launch_plask(window):
    dialog = LaunchDialog(window)
    global _launch_args
    result = dialog.exec_()
    _launch_args = dialog.args.text()
    if result == QtGui.QDialog.Accepted:
        launcher = LAUNCHERS[dialog.current]
        launcher.launch(window, *shlex.split(_launch_args))