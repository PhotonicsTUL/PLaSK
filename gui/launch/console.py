# coding utf:8

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
import subprocess

try:
    from shlex import quote
except ImportError:
    from pipes import quote

from .. import XPLDocument
from ..qt.QtGui import *
from ..qt.QtWidgets import *
from ..utils.config import CONFIG


def which(program):
    if os.path.split(program)[0]:
        if os.path.isfile(program) and os.access(program, os.X_OK):
            return program
    else:
        for path in [os.path.dirname(sys.executable)] + os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
                return exe_file
    return None


class Launcher(object):
    name = "Local Console"

    def __init__(self):
        self.dirname = None

    def widget(self, main_window, parent=None):
        widget = QWidget(parent)
        layout = QVBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(QLabel("Working directory:"))
        dirbutton = QPushButton()
        if self.dirname:
            dirname = self.dirname
        else:
            dirname = os.path.dirname(os.path.abspath(main_window.document.filename or u'dummy'))
        dirbutton.setIcon(QIcon.fromTheme('folder-open'))
        dirbutton.pressed.connect(lambda: self.select_workdir(main_window.document.filename))
        dirlayout = QHBoxLayout()
        self.diredit = QLineEdit()
        self.diredit.setReadOnly(True)
        self.diredit.setText(dirname)
        pal = self.diredit.palette()
        pal.setColor(QPalette.Base, QPalette().color(QPalette.Normal, QPalette.Window))
        self.diredit.setPalette(pal)
        dirlayout.addWidget(self.diredit)
        dirlayout.addWidget(dirbutton)
        layout.addLayout(dirlayout)
        layout.addWidget(QLabel("Log level:"))
        self.loglevel = QComboBox()
        loglevels = ["Error", "Warning", "Info", "Result", "Data", "Detail", "Debug"]
        self.loglevel.addItems(loglevels)
        if isinstance(main_window.document, XPLDocument):
            try:
                self.loglevel.setCurrentIndex(loglevels.index(main_window.document.loglevel.title()))
            except ValueError:
                self.loglevel.setCurrentIndex(5)
        else:
            self.loglevel.setCurrentIndex(5)
        layout.addWidget(self.loglevel)
        layout.setContentsMargins(1, 1, 1, 1)
        return widget

    def launch(self, main_window, args, defs):

        program = CONFIG['launcher_local/program']
        if not (program and os.path.isfile(program) and os.access(program, os.X_OK)):
            program = 'plask'
            if os.name == 'nt': program += '.exe'
            program = which(program) or program

        if main_window.isWindowModified():
            confirm = QMessageBox.question(main_window, "Unsaved File",
                                                 "The file must be saved before launching local computations. "
                                                 "Do you want to save the file now?",
                                                 QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.No or not main_window.save():
                return
        filename = os.path.abspath(main_window.document.filename)
        if self.dirname:
            dirname = self.dirname
        else:
            dirname = os.path.dirname(filename)

        loglevel = ("error_details", "warning", "info", "result", "data", "detail", "debug") \
                   [self.loglevel.currentIndex()]

        if CONFIG['workarounds/disable_omp']:
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '1'
        else:
            env = None

        script = os.path.dirname(__file__)

        if os.name == 'nt':
            script = os.path.join(script, 'console.bat')
            subprocess.Popen([script, program, '-l{}'.format(loglevel)] + list(defs) + ['--', filename] + list(args),
                             cwd=dirname, env=env)
        elif os.name == 'posix':
            script = os.path.join(script, 'console.sh')
            command = ' '.join(quote(s) for s in ['/bin/sh', script, program, '-l{}'.format(loglevel)] + list(defs) +
                               ['--', filename] + list(args))
            term = CONFIG['launcher_console/terminal']
            subprocess.Popen([term, '-e', command], cwd=dirname, env=env)
        else:
            raise NotImplemented("Launching terminal in this system")

    def select_workdir(self, filename):
        if self.dirname:
            dname = self.dirname
        else:
            dname = os.path.dirname(os.path.abspath(filename or 'dummy'))
        dirname = QFileDialog.getExistingDirectory(None, None, dname)
        if dirname:
            self.dirname = dirname
            self.diredit.setText(dirname)
