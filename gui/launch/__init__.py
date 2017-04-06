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
from ..qt.QtGui import *
from ..qt.QtWidgets import *
from ..qt.QtCore import *
from ..utils.config import CONFIG

from .local import Launcher as LocalLauncher
from .console import Launcher as ConsoleLauncher


_launch_args = ''
_defs_visible = False

LAUNCHERS = [LocalLauncher(), ConsoleLauncher()]

current_launcher = None


class LaunchDialog(QDialog):

    def __init__(self, window, parent=None):
        super(LaunchDialog, self).__init__(parent)
        self.setWindowTitle("Launch Computations")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        combo = QComboBox()
        combo.insertItems(len(LAUNCHERS), [item.name for item in LAUNCHERS])
        combo.currentIndexChanged.connect(self.launcher_changed, Qt.QueuedConnection)
        self.layout.addWidget(combo)

        if window.document.defines is not None:
            defines_layout = QHBoxLayout()
            defines_layout.setContentsMargins(0, 0, 0, 0)
            defines_button = QToolButton()
            defines_button.setIcon(QIcon.fromTheme('menu-down'))
            defines_button.setCheckable(True)
            defines_button.setChecked(_defs_visible)
            defines_button.toggled.connect(self.show_defines)
            self.defs_label = QLabel("Temporary de&fines:", self)
            self.defs_label.setBuddy(defines_button)
            defines_layout.addWidget(self.defs_label)
            defines_layout.addWidget(defines_button)
            self.layout.addLayout(defines_layout)

            self.defines = QPlainTextEdit()
            self.layout.addWidget(self.defines)
            self.defines.setVisible(_defs_visible)

            self.defines.setPlainText('\n'.join(e.name+'=' for e in window.document.defines.model.entries))

        self.args = QLineEdit()
        self.args.setText(_launch_args)
        args_label = QLabel("Command line &arguments:", self)
        args_label.setBuddy(self.args)
        self.layout.addWidget(args_label)
        self.layout.addWidget(self.args)

        self.launcher_widgets = [l.widget(window, self) for l in LAUNCHERS]
        global current_launcher
        if current_launcher is None:
            current_launcher = combo.findText(CONFIG['launcher/default'])
            if current_launcher == -1:
                current_launcher = combo.findText(LocalLauncher.name)
        for i, widget in enumerate(self.launcher_widgets):
            widget.setVisible(i == current_launcher)
            self.layout.addWidget(widget)

        combo.setCurrentIndex(current_launcher)

        self.setFixedWidth(5*QFontMetrics(QFont()).width(self.windowTitle()))
        self.setFixedHeight(self.sizeHint().height())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok |  QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout.addWidget(buttons)

        self.setFixedHeight(self.sizeHint().height())
        self.adjustSize()

    def show_defines(self, visible):
        global _defs_visible
        _defs_visible = visible
        self.defines.setVisible(visible)
        self.setFixedHeight(self.sizeHint().height())
        self.adjustSize()

    def launcher_changed(self, index):
        global current_launcher
        self.launcher_widgets[current_launcher].setVisible(False)
        current_launcher = index
        self.launcher_widgets[current_launcher].setVisible(True)
        self.setFixedHeight(self.sizeHint().height())
        self.adjustSize()


def launch_plask(window):
    dialog = LaunchDialog(window)
    global _launch_args
    result = dialog.exec_()
    launcher = LAUNCHERS[current_launcher]
    _launch_args = dialog.args.text()
    if result == QDialog.Accepted:
        launch_defs = []
        if window.document.defines is not None:
            for line in dialog.defines.toPlainText().split('\n'):
                if not line.strip(): continue
                if '=' not in line or line.startswith('-'):
                    msgbox = QMessageBox()
                    msgbox.setWindowTitle("Wrong Defines")
                    msgbox.setText("Wrong define: '{}'".format(line))
                    msgbox.setStandardButtons(QMessageBox.Ok)
                    msgbox.setIcon(QMessageBox.Critical)
                    msgbox.exec_()
                    return
                items = line.split('=')
                name = items[0].strip()
                value = '='.join(items[1:]).strip()
                if value:
                    launch_defs.append('{}={}'.format(name, value))
        launcher.launch(window, shlex.split(_launch_args), launch_defs)
    for launch in LAUNCHERS:
        try:
            launch.exit(launch is launcher)
        except AttributeError:
            pass
