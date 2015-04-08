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
import re
import subprocess
from time import strftime

from gui.qt import QtCore, QtGui
from gui.qt.QtCore import Qt

from gui.launch import LAUNCHERS
from gui.utils.config import CONFIG


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


class OutputWindow(QtGui.QDockWidget):

    def __init__(self, launcher, parent=None):
        super(OutputWindow, self).__init__("Launch local [{}]".format(strftime('%X')), parent)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.launcher = launcher

        font = QtGui.QFont()
        font_family = CONFIG['launcher_local/font_family']
        if font_family is None:
            if sys.platform == 'win32':
                font_family = "Consolas"
            elif sys.platform == 'darwin':
                font_family = "Monaco"
            else:
                font_family = "Monospace"
            CONFIG['launcher_local/font_fpycodeamily'] = font_family
            font.setStyleHint(QtGui.QFont.TypeWriter)
        font.setFamily(font_family)
        font.setPointSize(int(CONFIG('launcher_local/font_size', 10)))
        self.messages = QtGui.QTextBrowser()
        self.messages.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.messages.setReadOnly(True)
        self.messages.setAcceptRichText(True)
        self.messages.setFont(font)

        self.messages.anchorClicked.connect(self.url_clicked)
        self.messages.setOpenLinks(False)

        toolbar = QtGui.QToolBar()

        self.action_error = QtGui.QAction(self)
        self.action_error.setText("&Error")
        self.action_error.setCheckable(True)
        self.action_error.setChecked(launcher.error.isChecked())
        self.action_error.triggered.connect(self.update_view)
        self.action_error.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_error.setShortcut('1')
        self.action_warning = QtGui.QAction(self)
        self.action_warning.setText("&Warning")
        self.action_warning.setCheckable(True)
        self.action_warning.setChecked(launcher.warning.isChecked())
        self.action_warning.triggered.connect(self.update_view)
        self.action_warning.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_warning.setShortcut('2')
        self.action_info = QtGui.QAction(self)
        self.action_info.setText("&Info")
        self.action_info.setCheckable(True)
        self.action_info.setChecked(launcher.info.isChecked())
        self.action_info.triggered.connect(self.update_view)
        self.action_info.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_info.setShortcut('3')
        self.action_result = QtGui.QAction(self)
        self.action_result.setText("&Result")
        self.action_result.setCheckable(True)
        self.action_result.setChecked(launcher.result.isChecked())
        self.action_result.triggered.connect(self.update_view)
        self.action_result.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_result.setShortcut('4')
        self.action_data = QtGui.QAction(self)
        self.action_data.setText("&Data")
        self.action_data.setCheckable(True)
        self.action_data.setChecked(launcher.data.isChecked())
        self.action_data.triggered.connect(self.update_view)
        self.action_data.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_data.setShortcut('5')
        self.action_detail = QtGui.QAction(self)
        self.action_detail.setText("De&tail")
        self.action_detail.setCheckable(True)
        self.action_detail.setChecked(launcher.detail.isChecked())
        self.action_detail.triggered.connect(self.update_view)
        self.action_detail.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_detail.setShortcut('6')
        self.action_debug = QtGui.QAction(self)
        self.action_debug.setText("De&bug")
        self.action_debug.setCheckable(True)
        self.action_debug.setChecked(launcher.debug.isChecked())
        self.action_debug.triggered.connect(self.update_view)
        self.action_debug.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_debug.setShortcut('7')

        self.addAction(self.action_error)
        self.addAction(self.action_warning)
        self.addAction(self.action_info)
        self.addAction(self.action_result)
        self.addAction(self.action_data)
        self.addAction(self.action_detail)
        self.addAction(self.action_debug)

        view_menu = QtGui.QMenu("Show")
        view_menu.addAction(self.action_error)
        view_menu.addAction(self.action_warning)
        view_menu.addAction(self.action_info)
        view_menu.addAction(self.action_result)
        view_menu.addAction(self.action_data)
        view_menu.addAction(self.action_detail)
        view_menu.addAction(self.action_debug)

        menu_button = QtGui.QToolButton()
        menu_button.setMenu((view_menu))
        menu_button.setPopupMode(QtGui.QToolButton.InstantPopup)
        menu_button.setIcon(QtGui.QIcon.fromTheme('edit-find'))
        menu_button.setFocusPolicy(Qt.NoFocus)
        # menu_button.setText("Log Levels")
        tool_action = QtGui.QWidgetAction(self)
        tool_action.setDefaultWidget(menu_button)
        toolbar.addAction(tool_action)

        self.halt_action = QtGui.QAction(QtGui.QIcon.fromTheme('process-stop'),
                                         "Halt (Alt+X)", self)
        self.halt_action.setShortcut('Alt+x')
        self.halt_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.messages.addAction(self.halt_action)
        toolbar.addAction(self.halt_action)

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.messages)

        widget = QtGui.QWidget(self)
        widget.setLayout(layout)
        self.setWidget(widget)

        close_action = QtGui.QAction(self)
        close_action.setShortcut('Ctrl+w')
        close_action.triggered.connect(self.close)
        close_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(close_action)

        self.lines = []
        self.printed_lines = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_output)
        self.timer.start(250)

        self.visibilityChanged.connect(self.on_visibility_changed)
        self.messages.setFocus()

    def url_clicked(self, url):
        parent = self.parent()
        if parent:
            parent.goto_line(int(url.path()))

    def update_view(self):
        self.messages.clear()
        self.printed_lines = 0
        self.update_output()

    def update_output(self):
        move = self.messages.verticalScrollBar().value() == self.messages.verticalScrollBar().maximum()
        try:
            self.launcher.mutex.lock()
            total_lines = len(self.lines)
            lines = []
            if self.printed_lines != total_lines:
                for cat, line in self.lines[self.printed_lines:total_lines]:
                    if 'ERROR' in cat and not self.action_error.isChecked(): continue
                    if cat == 'WARNING' and not self.action_warning.isChecked(): continue
                    if cat == 'INFO' and not self.action_info.isChecked(): continue
                    if cat == 'RESULT' and not self.action_result.isChecked(): continue
                    if cat == 'DATA' and not self.action_data.isChecked(): continue
                    if cat == 'DETAIL' and not self.action_detail.isChecked(): continue
                    if cat == 'DEBUG' and not self.action_debug.isChecked(): continue
                    lines.append(line)
                if lines:
                    self.messages.append(u"<br/>\n".join(lines))
                self.printed_lines = total_lines
            else:
                move = False
        finally:
            self.launcher.mutex.unlock()
        if move:
            self.messages.moveCursor(QtGui.QTextCursor.End)

    def halt_thread(self):
        confirm = QtGui.QMessageBox.question(self, "Halt Process",
                                             "PLaSK is currently running. Do you really want to terminate it? "
                                             "All computation results may be lost!",
                                             QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if confirm == QtGui.QMessageBox.Yes:
            self.thread.kill_process()

    def thread_finished(self):
        self.setWindowTitle(self.windowTitle() + " ({})".format(strftime('%X')))
        self.halt_action.setEnabled(False)

    def on_visibility_changed(self, visible):
        if visible:
            self.messages.setFocus()

    def closeEvent(self, event):
        focus = self.messages.hasFocus()
        if self.thread.isRunning():
            confirm = QtGui.QMessageBox.question(self, "Close Window",
                                                 "PLaSK process is currently running. Closing the window "
                                                 "will terminate it. Do you really want to proceed? "
                                                 "All computation results may be lost!",
                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if confirm == QtGui.QMessageBox.Yes:
                self.thread.kill_process()
                if not self.thread.wait(6000):
                    QtGui.QMessageBox.critical(self, "Close Window",
                                               "PLaSK process could not be terminated. Window will not be closed. "
                                               "Please try once again or contact the program authors.",
                                               QtGui.QMessageBox.Ok)
                    event.ignore()
                    return
            else:
                event.ignore()
                return
        super(OutputWindow, self).closeEvent(event)
        if focus:
            main_window = self.parent()
            others = [w for w in main_window.findChildren(QtGui.QDockWidget)
                      if isinstance(w, OutputWindow) and w is not self and w.isVisible()]
            if others:
                others[-1].messages.setFocus()


class PlaskThread(QtCore.QThread):

    def __init__(self, program, fname, dirname, lines, mutex, main_window, *args):
        super(PlaskThread, self).__init__()
        self.main_window = main_window
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags = subprocess.STARTF_USESTDHANDLES | subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
        except AttributeError:
            self.proc = subprocess.Popen([program, '-ldebug', '-u', fname] + list(args),
                                         cwd=dirname, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            self.proc = subprocess.Popen([program, '-ldebug', '-u', '-w', fname] + list(args), startupinfo=si,
                                         cwd=dirname, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.link = re.compile(r'({}(?:,|:&nbsp;XMLError:&nbsp;XML))&nbsp;line&nbsp;(\d+)'.format(fname))
        self.lines = lines
        self.mutex = mutex
        self.terminated.connect(self.kill_process)
        self.main_window.closed.connect(self.kill_process)

    def __del__(self):
        self.main_window.closed.disconnect(self.kill_process)

    def parse_line(self, line):
        if not line: return
        cat = line[:15]
        line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if   cat == "CRITICAL ERROR:": color = "red    "
        elif cat == "ERROR         :": color = "red    "
        elif cat == "WARNING       :": color = "brown  "
        elif cat == "INFO          :": color = "blue   "
        elif cat == "RESULT        :": color = "green  "
        elif cat == "DATA          :": color = "#006060"
        elif cat == "DETAIL        :": color = "black  "
        elif cat == "ERROR DETAIL  :": color = "#800000"
        elif cat == "DEBUG         :": color = "gray   "
        else: color = "black; font-weight:bold"
        line = line.replace(' ', '&nbsp;')
        line = self.link.sub(r'\1&nbsp;<a href="line:\2">line&nbsp;\2</a>', line)
        try:
            line = line.decode(self.main_window.document.coding)
        except UnicodeDecodeError:
            line = line.decode('utf-8')
        try:
            self.mutex.lock()
            self.lines.append((cat[:-1].strip(),
                               u'<span style="color:{};">{}</span>'.format(color, line)))
        finally:
            self.mutex.unlock()

    def run(self):
        while self.proc.poll() is None:
            line = self.proc.stdout.readline().rstrip()
            self.parse_line(line)
        out, _ = self.proc.communicate()
        for line in out.splitlines():
            self.parse_line(line)

    def kill_process(self):
        try:
            self.proc.terminate()
        except OSError:
            pass


class Launcher(object):
    name = "Local Process"

    def __init__(self):
        self.dirname = None
        self.program = CONFIG['launcher_local/program']
        if not (self.program and os.path.isfile(self.program) and os.access(self.program, os.X_OK)):
            self.program = which('plask') or 'plask'

    def widget(self, main_window):
        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        widget.setLayout(layout)
        layout.addWidget(QtGui.QLabel("Working directory:"))
        dirbutton = QtGui.QPushButton()
        if self.dirname:
            dirname = self.dirname
        else:
            dirname = os.path.dirname(os.path.abspath(main_window.document.filename or 'dummy'))
        dirbutton.setIcon(QtGui.QIcon.fromTheme('folder-open'))
        dirbutton.pressed.connect(lambda: self.select_workdir(main_window.document.filename))
        dirlayout = QtGui.QHBoxLayout()
        self.diredit = QtGui.QLineEdit()
        self.diredit.setReadOnly(True)
        self.diredit.setText(dirname)
        pal = self.diredit.palette()
        pal.setColor(QtGui.QPalette.Base, QtGui.QPalette().color(QtGui.QPalette.Normal, QtGui.QPalette.Window))
        self.diredit.setPalette(pal)
        dirlayout.addWidget(self.diredit)
        dirlayout.addWidget(dirbutton)
        layout.addLayout(dirlayout)
        layout.addWidget(QtGui.QLabel("Default log levels:"))
        self.error = QtGui.QCheckBox("&Error")
        self.error.setChecked(int(CONFIG('launcher_local/show_error', 2)) == 2)
        self.error.stateChanged.connect(lambda state: CONFIG.__setitem__('launcher_local/show_error', state))
        layout.addWidget(self.error)
        self.warning = QtGui.QCheckBox("&Warning")
        self.warning.setChecked(int(CONFIG('launcher_local/show_warning', 2)) == 2)
        layout.addWidget(self.warning)
        self.warning.stateChanged.connect(lambda state: CONFIG.__setitem__('launcher_local/show_warning', state))
        self.info = QtGui.QCheckBox("&Info")
        self.info.setChecked(int(CONFIG('launcher_local/show_info', 2)) == 2)
        layout.addWidget(self.info)
        self.info.stateChanged.connect(lambda state: CONFIG.__setitem__('launcher_local/show_info', state))
        self.result = QtGui.QCheckBox("&Result")
        self.result.setChecked(int(CONFIG('launcher_local/show_result', 2)) == 2)
        layout.addWidget(self.result)
        self.result.stateChanged.connect(lambda state: CONFIG.__setitem__('launcher_local/show_result', state))
        self.data = QtGui.QCheckBox("&Data")
        self.data.setChecked(int(CONFIG('launcher_local/show_data', 2)) == 2)
        layout.addWidget(self.data)
        self.data.stateChanged.connect(lambda state: CONFIG.__setitem__('launcher_local/show_data', state))
        self.detail = QtGui.QCheckBox("De&tail")
        self.detail.setChecked(int(CONFIG('launcher_local/show_detail', 2)) == 2)
        layout.addWidget(self.detail)
        self.detail.stateChanged.connect(lambda state: CONFIG.__setitem__('launcher_local/show_detail', state))
        self.debug = QtGui.QCheckBox("De&bug")
        self.debug.setChecked(int(CONFIG('launcher_local/show_debug', 2)) == 2)
        self.debug.stateChanged.connect(lambda state: CONFIG.__setitem__('launcher_local/show_debug', state))
        layout.addWidget(self.debug)
        layout.setContentsMargins(1, 1, 1, 1)
        return widget

    def launch(self, main_window, *args):

        if main_window.isWindowModified():
            confirm = QtGui.QMessageBox.question(main_window, "Unsaved File",
                                                 "The file must be saved before launching local computations. "
                                                 "Do you want to save the file now?",
                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if confirm == QtGui.QMessageBox.No or not main_window.save():
                return
        filename = os.path.abspath(main_window.document.filename)
        if self.dirname:
            dirname = self.dirname
        else:
            dirname = os.path.dirname(filename)
            
        dock = OutputWindow(self, main_window)
        try:
            bottom_docked = [w for w in main_window.findChildren(QtGui.QDockWidget)
                             if main_window.dockWidgetArea(w) == (Qt.BottomDockWidgetArea)][-1]
        except IndexError:
            main_window.addDockWidget(Qt.BottomDockWidgetArea, dock)
        else:
            main_window.addDockWidget(Qt.BottomDockWidgetArea, dock)
            main_window.tabifyDockWidget(bottom_docked, dock)
            dock.show()
            dock.raise_()

        self.mutex = QtCore.QMutex()
        dock.thread = PlaskThread(self.program, filename, dirname, dock.lines, self.mutex, main_window, *args)
        dock.thread.finished.connect(dock.thread_finished)
        dock.halt_action.triggered.connect(dock.halt_thread)
        dock.thread.start()

    def select_workdir(self, filename):
        if self.dirname:
            dname = self.dirname
        else:
            dname = os.path.dirname(os.path.abspath(filename or 'dummy'))
        dirname = QtGui.QFileDialog.getExistingDirectory(None, None, dname)
        if dirname:
            self.dirname = dirname
            self.diredit.setText(dirname)


LAUNCHERS.insert(0, Launcher())

