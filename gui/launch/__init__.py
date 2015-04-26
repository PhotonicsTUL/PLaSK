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

from ..qt import QtGui
from ..qt.QtCore import Qt

from .local import Launcher as LocalLauncher


_launch_args = ''

LAUNCHERS = [LocalLauncher()]


class LaunchDialog(QtGui.QDialog):

    def __init__(self, window, parent=None):
        super(LaunchDialog, self).__init__(parent)
        self.setWindowTitle("Launch Computations")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)

        combo = QtGui.QComboBox()
        combo.insertItems(len(LAUNCHERS), [item.name for item in LAUNCHERS])
        combo.currentIndexChanged.connect(self.launcher_changed)
        self.layout.addWidget(combo)

        defines_layout = QtGui.QHBoxLayout()
        defines_layout.setContentsMargins(0, 0, 0, 0)
        defines_button = QtGui.QToolButton()
        defines_button.setIcon(QtGui.QIcon.fromTheme('menu-down'))
        defines_button.setCheckable(True)
        defines_button.toggled.connect(self.show_defines)
        self.defs_label = QtGui.QLabel("Temporary de&fines:", self)
        self.defs_label.setBuddy(defines_button)
        defines_layout.addWidget(self.defs_label)
        defines_layout.addWidget(defines_button)
        self.layout.addLayout(defines_layout)

        self.defines = QtGui.QPlainTextEdit()
        self.layout.addWidget(self.defines)
        self.defines.setVisible(False)

        if window.document.defines is not None:
            self.defines.setPlainText('\n'.join(e.name+'=' for e in window.document.defines.model.entries))

        self.args = QtGui.QLineEdit()
        self.args.setText(_launch_args)
        args_label = QtGui.QLabel("Command line &arguments:", self)
        args_label.setBuddy(self.args)
        self.layout.addWidget(args_label)
        self.layout.addWidget(self.args)

        self.launcher_widgets = [l.widget(window) for l in LAUNCHERS]
        for widget in self.launcher_widgets:
            widget.setVisible(False)
            self.layout.addWidget(widget)
        self.current = combo.currentIndex()
        self.launcher_widgets[self.current].setVisible(True)

        self.setFixedWidth(6*QtGui.QFontMetrics(QtGui.QFont()).width(self.windowTitle()))
        self.setFixedHeight(self.sizeHint().height())

        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui. QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

    def show_defines(self, visible):
        self.defines.setVisible(visible)
        self.setFixedHeight(self.sizeHint().height())
        self.adjustSize()

    def launcher_changed(self, index):
        self.launcher_widgets[self.current].setVisible(False)
        self.current = index
        self.launcher_widgets[self.current].setVisible(True)
        self.setFixedHeight(self.sizeHint().height())
        self.adjustSize()


def launch_plask(window):
    dialog = LaunchDialog(window)
    global _launch_args
    result = dialog.exec_()
    _launch_args = dialog.args.text()
    launch_defs = []
    for line in dialog.defines.toPlainText().split('\n'):
        if not line.strip(): continue
        if '=' not in line or line.startswith('-'):
            msgbox = QtGui.QMessageBox()
            msgbox.setWindowTitle("Wrong Defines")
            msgbox.setText("Wrong define: '{}'".format(line))
            msgbox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgbox.setIcon(QtGui.QMessageBox.Critical)
            msgbox.exec_()
            return
        items = line.split('=')
        name = items[0].strip()
        value = '='.join(items[1:]).strip()
        if value:
            launch_defs.append('{}={}'.format(name, value))
    if result == QtGui.QDialog.Accepted:
        launcher = LAUNCHERS[dialog.current]
        launcher.launch(window, shlex.split(_launch_args), launch_defs)