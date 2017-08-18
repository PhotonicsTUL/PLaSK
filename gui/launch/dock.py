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
from time import strftime

from ..qt.QtCore import Qt, QTimer
from ..qt.QtGui import QFont, QTextOption, QIcon, QTextCursor
from ..qt.QtWidgets import QDockWidget, QTextBrowser, QToolBar, QAction, QMenu, QToolButton, QWidgetAction, \
    QVBoxLayout, QWidget, QMessageBox

from ..utils.config import CONFIG


class OutputWindow(QDockWidget):

    def __init__(self, launcher, main_window, label="Launch local"):
        super(OutputWindow, self).__init__("{} [{}]".format(label, strftime('%X')), main_window)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.launcher = launcher
        self.main_window = main_window

        if main_window is not None:
            main_window.closing.connect(self.check_close_event)

        font = QFont()
        font.setStyleHint(QFont.TypeWriter)
        font.fromString(','.join(CONFIG['launcher_local/font']))
        self.messages = QTextBrowser()
        self.messages.setWordWrapMode(QTextOption.NoWrap)
        self.messages.setReadOnly(True)
        self.messages.setAcceptRichText(True)
        self.messages.setFont(font)

        self.messages.anchorClicked.connect(self.url_clicked)
        self.messages.setOpenLinks(False)

        toolbar = QToolBar()

        self.action_error = QAction(self)
        self.action_error.setText("&Error")
        self.action_error.setCheckable(True)
        self.action_error.setChecked(launcher.error.isChecked())
        self.action_error.triggered.connect(self.update_view)
        self.action_error.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_error.setShortcut('1')
        self.action_warning = QAction(self)
        self.action_warning.setText("&Warning")
        self.action_warning.setCheckable(True)
        self.action_warning.setChecked(launcher.warning.isChecked())
        self.action_warning.triggered.connect(self.update_view)
        self.action_warning.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_warning.setShortcut('2')
        self.action_info = QAction(self)
        self.action_info.setText("&Info")
        self.action_info.setCheckable(True)
        self.action_info.setChecked(launcher.info.isChecked())
        self.action_info.triggered.connect(self.update_view)
        self.action_info.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_info.setShortcut('3')
        self.action_result = QAction(self)
        self.action_result.setText("&Result")
        self.action_result.setCheckable(True)
        self.action_result.setChecked(launcher.result.isChecked())
        self.action_result.triggered.connect(self.update_view)
        self.action_result.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_result.setShortcut('4')
        self.action_data = QAction(self)
        self.action_data.setText("&Data")
        self.action_data.setCheckable(True)
        self.action_data.setChecked(launcher.data.isChecked())
        self.action_data.triggered.connect(self.update_view)
        self.action_data.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_data.setShortcut('5')
        self.action_detail = QAction(self)
        self.action_detail.setText("De&tail")
        self.action_detail.setCheckable(True)
        self.action_detail.setChecked(launcher.detail.isChecked())
        self.action_detail.triggered.connect(self.update_view)
        self.action_detail.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_detail.setShortcut('6')
        self.action_debug = QAction(self)
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

        view_menu = QMenu("Show")
        view_menu.addAction(self.action_error)
        view_menu.addAction(self.action_warning)
        view_menu.addAction(self.action_info)
        view_menu.addAction(self.action_result)
        view_menu.addAction(self.action_data)
        view_menu.addAction(self.action_detail)
        view_menu.addAction(self.action_debug)

        menu_button = QToolButton()
        menu_button.setMenu((view_menu))
        menu_button.setPopupMode(QToolButton.InstantPopup)
        menu_button.setIcon(QIcon.fromTheme('edit-find'))
        menu_button.setFocusPolicy(Qt.NoFocus)
        # menu_button.setText("Log Levels")
        tool_action = QWidgetAction(self)
        tool_action.setDefaultWidget(menu_button)
        toolbar.addAction(tool_action)

        self.halt_action = QAction(QIcon.fromTheme('process-stop'),
                                         "Halt (Alt+X)", self)
        self.halt_action.setShortcut('Alt+x')
        self.halt_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.halt_action.triggered.connect(self.halt_thread)
        self.messages.addAction(self.halt_action)
        toolbar.addAction(self.halt_action)

        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.messages)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setWidget(widget)

        close_action = QAction(self)
        close_action.setShortcut('Ctrl+w')
        close_action.triggered.connect(self.close)
        close_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.addAction(close_action)

        self.lines = []
        self.printed_lines = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_output)
        self.timer.start(250)

        self.visibilityChanged.connect(self.on_visibility_changed)
        self.messages.setFocus()

        try:
            main_window.config_changed.connect(self.reconfig)
        except AttributeError:
            pass

        self.thread = None

    def reconfig(self):
        font = self.messages.font()
        if font.fromString(','.join(CONFIG['launcher_local/font'])):
            self.messages.setFont(font)

    def url_clicked(self, url):
        parent = self.parent()
        if parent:
            parent.goto_line(int(url.path()))

    def update_view(self):
        self.messages.clear()
        self.printed_lines = 0
        self.update_output()

    def parse_line(self, line, link=None):
        if not line: return
        try:
            line = line.decode(self.main_window.document.coding)
        except UnicodeDecodeError:
            line = line.decode('utf-8')
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
        if link is not None:
            line = link.sub(u'<a style="color: {}; text-decoration: none;" href="line:\\2">\\1\\2\\3</a>'.format(color),
                            line)
        try:
            self.launcher.mutex.lock()
            self.lines.append((cat[:-1].strip(),
                               u'<span style="color:{};">{}</span>'.format(color, line)))
        finally:
            self.launcher.mutex.unlock()

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
            hpos = self.messages.horizontalScrollBar().value()
            self.messages.moveCursor(QTextCursor.End)
            self.messages.horizontalScrollBar().setValue(hpos)

    def halt_thread(self):
        confirm = QMessageBox.question(self, "Halt Process",
                                             "PLaSK is currently running. Do you really want to terminate it? "
                                             "All computation results may be lost!",
                                             QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes and self.thread is not None:
            self.thread.kill_process()

    def thread_finished(self):
        self.setWindowTitle(self.windowTitle() + " ({})".format(strftime('%X')))
        self.halt_action.setEnabled(False)

    def on_visibility_changed(self, visible):
        if visible:
            self.messages.setFocus()

    def check_close_event(self, event):
        checked = getattr(event, 'checked_by_laucher_local', False)
        if not checked and event.isAccepted() and self.thread is not None and self.thread.isRunning():
            event.checked_by_laucher_local = True
            confirm = QMessageBox.question(self, "Close Window",
                                                 "PLaSK process is currently running. Closing the window "
                                                 "will terminate it. Do you really want to proceed? "
                                                 "All computation results may be lost!",
                                                 QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                self.thread.kill_process()
                if not self.thread.wait(6000):
                    QMessageBox.critical(self, "Close Window",
                                               "PLaSK process could not be terminated. Window will not be closed. "
                                               "Please try once again or contact the program authors.",
                                               QMessageBox.Ok)
                    event.ignore()
                    event.ignore()
            else:
                event.ignore()

    def closeEvent(self, event):
        focus = self.messages.hasFocus()
        self.check_close_event(event)
        if not event.isAccepted():
            return
        super(OutputWindow, self).closeEvent(event)
        if focus:
            main_window = self.parent()
            others = [w for w in main_window.findChildren(QDockWidget)
                      if isinstance(w, OutputWindow) and w is not self and w.isVisible()]
            if others:
                others[-1].messages.setFocus()