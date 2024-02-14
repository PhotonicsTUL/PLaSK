# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# coding utf:8


import os
import subprocess
import sys

from .dock import OutputWindow
from ..qt.QtCore import *
from ..qt.QtGui import *
from ..qt.QtWidgets import *
from ..utils.config import CONFIG
from ..utils.files import which


class LaunchThread(QThread):

    def __init__(self, program, fname, dirname, dock, main_window, args, defs):
        super().__init__()
        self.main_window = main_window
        env = os.environ.copy()
        if CONFIG['workarounds/disable_omp']:
            env['OMP_NUM_THREADS'] = '1'
        env['PYTHONIOENCODING'] = self.main_window.document.coding
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags = subprocess.STARTF_USESTDHANDLES | subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
        except AttributeError:
            self.proc = subprocess.Popen([program, '-ldebug', '-g'] + list(defs) + ['--', fname] + list(args),
                                         cwd=dirname, stdout=subprocess.PIPE, env=env, stderr=subprocess.STDOUT,
                                         bufsize=0)
        else:
            self.proc = subprocess.Popen([program, '-ldebug', '-g'] + list(defs) + ['--', fname] + list(args),
                                         cwd=dirname, stdout=subprocess.PIPE, env=env, stderr=subprocess.STDOUT,
                                         bufsize=0, startupinfo=si)
        self.dock = dock
        try:
            self.terminated.connect(self.kill_process)
        except AttributeError:
            self.finished.connect(self.kill_process)
        self.main_window.closed.connect(self.kill_process)

    def __del__(self):
        try:
            self.main_window.closed.disconnect(self.kill_process)
        except:
            pass

    def run(self):
        while self.proc.poll() is None:
            line = self.proc.stdout.readline().rstrip()
            self.dock.parse_line(line)
        out, _ = self.proc.communicate()
        for line in out.splitlines():
            self.dock.parse_line(line)

    def kill_process(self):
        try:
            self.proc.terminate()
        except OSError:
            pass


class Launcher:
    name = "Local Process"

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
        pal.setColor(QPalette.ColorRole.Base, QPalette().color(QPalette.ColorGroup.Normal, QPalette.ColorRole.Window))
        self.diredit.setPalette(pal)
        dirlayout.addWidget(self.diredit)
        dirlayout.addWidget(dirbutton)
        layout.addLayout(dirlayout)

        layout.addWidget(QLabel("Visible Log levels:"))
        try:
            loglevel = ['error', 'warning', 'important', 'info', 'result', 'data', 'detail', 'debug'].index(
                main_window.document.loglevel.lower())
        except AttributeError:
            loglevel = 6
        self.error = QCheckBox("&Error")
        self.error.setChecked(loglevel >= 0)
        layout.addWidget(self.error)
        self.warning = QCheckBox("&Warning")
        self.warning.setChecked(loglevel >= 1)
        layout.addWidget(self.warning)
        self.important = QCheckBox("I&mportant")
        self.important.setChecked(loglevel >= 2)
        layout.addWidget(self.important)
        self.info = QCheckBox("&Info")
        self.info.setChecked(loglevel >= 3)
        layout.addWidget(self.info)
        self.result = QCheckBox("&Result")
        self.result.setChecked(loglevel >= 4)
        layout.addWidget(self.result)
        self.data = QCheckBox("&Data")
        self.data.setChecked(loglevel >= 5)
        layout.addWidget(self.data)
        self.detail = QCheckBox("De&tail")
        self.detail.setChecked(loglevel >= 6)
        layout.addWidget(self.detail)
        self.debug = QCheckBox("De&bug")
        self.debug.setChecked(loglevel >= 7)
        layout.addWidget(self.debug)

        layout.setContentsMargins(1, 1, 1, 1)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.No or not main_window.save():
                return
        filename = os.path.abspath(main_window.document.filename)
        if self.dirname:
            dirname = self.dirname
        else:
            dirname = os.path.dirname(filename)

        dock = OutputWindow(self, main_window)
        try:
            bottom_docked = [w for w in main_window.findChildren(QDockWidget)
                             if main_window.dockWidgetArea(w) == (Qt.DockWidgetArea.BottomDockWidgetArea)][-1]
        except IndexError:
            main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        else:
            main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            main_window.tabifyDockWidget(bottom_docked, dock)
            dock.show()
            dock.raise_()

        dock.thread = LaunchThread(program, filename, dirname, dock, main_window, args, defs)
        dock.thread.finished.connect(dock.thread_finished)
        dock.thread.start()

    def select_workdir(self, filename):
        if self.dirname:
            dname = self.dirname
        else:
            dname = os.path.dirname(os.path.abspath(filename or 'dummy'))
        dirname = QFileDialog.getExistingDirectory(None, None, dname)
        if dirname:
            self.dirname = dirname
            self.diredit.setText(dirname)
