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

import os.path
import re
from time import strftime

from ..utils.widgets import LineEditWithClear, set_icon_size

from ..qt.QtCore import *
from ..qt.QtGui import *
from ..qt.QtWidgets import *
from ..qt import qt_exec

from ..utils.config import CONFIG, set_font

LEVEL_CRITICAL_ERROR = 1
LEVEL_ERROR          = 2
LEVEL_WARNING        = 3
LEVEL_IMPORTANT      = 4
LEVEL_INFO           = 5
LEVEL_RESULT         = 6
LEVEL_DATA           = 7
LEVEL_DETAIL         = 8
LEVEL_ERROR_DETAIL   = 9
LEVEL_DEBUG          = 10

LEVEL_ROLE = Qt.ItemDataRole.UserRole
LINE_ROLE = Qt.ItemDataRole.UserRole + 1

class OutputModel(QAbstractListModel):

    def __init__(self, fm):
        super().__init__()
        self.update_font(fm)
        self.lines = []

    def update_font(self, fm):
        self.fm = fm
        if fm is not None:
            self.lh = fm.lineSpacing()
            self.lw = fm.maxWidth()

    def add_line(self, level, text, link=None):
        ll = len(self.lines)
        self.beginInsertRows(QModelIndex(), ll, ll+1)
        self.lines.append((level, text, link))
        self.endInsertRows()

    def data(self, index, role=None):
        if not index.isValid():
            return
        row = index.row()
        if role == Qt.ItemDataRole.DisplayRole:
            return self.lines[row][1]
        if role == Qt.ItemDataRole.ForegroundRole:
            level = self.lines[row][0]
            color = CONFIG['launcher_local/color_{:d}'.format(level)]
            return QBrush(QColor(color))
        if role == Qt.ItemDataRole.BackgroundRole:
            color = CONFIG['launcher_local/background_color']
            return QBrush(QColor(color))
        if role == LEVEL_ROLE:
            return self.lines[row][0]
        if role == LINE_ROLE:
            return self.lines[row][2]
        if role == Qt.ItemDataRole.SizeHintRole and self.fm is not None:
            return QSize(self.fm.horizontalAdvance(self.lines[row][1])+self.lw, self.lh)

    def rowCount(self, parent=None):
        return len(self.lines)

    # def columnCount(self, parent=None):
    #     return 2


class OutputListView(QListView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.context_menu = QMenu()
        copy_action = QAction(QIcon.fromTheme('edit-copy'), "&Copy", self)
        CONFIG.set_shortcut(copy_action, 'launcher_copy')
        copy_action.triggered.connect(self.copy)
        self.context_menu.addAction(copy_action)
        self.addAction(copy_action)
        self.context_menu.addSeparator()
        select_all_action = QAction("Select &All", self)
        CONFIG.set_shortcut(select_all_action, 'launcher_select_all')
        select_all_action.triggered.connect(self.selectAll)
        self.context_menu.addAction(select_all_action)
        clear_selection_action = QAction("Clea&r Selection", self)
        CONFIG.set_shortcut(clear_selection_action, 'launcher_clear_selection')
        clear_selection_action.triggered.connect(self.clearSelection)
        self.context_menu.addAction(clear_selection_action)

        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Base, QColor(CONFIG['launcher_local/background_color']))
        pal.setColor(QPalette.ColorRole.Text, QColor(CONFIG['launcher_local/color_0']))
        self.setPalette(pal)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        index = self.indexAt(event.pos())
        line = self.model().data(index, LINE_ROLE)
        if line is not None:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def contextMenuEvent(self, event):
        qt_exec(self.context_menu, event.globalPos())

    def copy(self):
        rows = self.selectionModel().selectedRows()
        rows.sort(key=lambda row: row.row())
        lines = [self.model().data(row, Qt.ItemDataRole.DisplayRole) for row in rows]
        QApplication.clipboard().setText('\n'.join(lines))


class OutputFilter(QSortFilterProxyModel):

    def __init__(self, window, model):
        super().__init__()
        self.window = window
        self.setSourceModel(model)

    def filterAcceptsRow(self, row, parent):
        try:
            level = self.sourceModel().lines[row][0]
        except IndexError:
            return False
        if level != 0 and not self.window.levels[level-1].isChecked():
            return False
        return super().filterAcceptsRow(row, parent)


class OutputWindow(QDockWidget):

    def __init__(self, launcher, main_window, filename=None, label="Launch local"):
        super().__init__("{} [{}]".format(label, strftime('%X')), main_window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self.launcher = launcher
        self.main_window = main_window

        if main_window is not None:
            main_window.closing.connect(self.check_close_event)

        font = QFont()
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        set_font(font, 'launcher_local/font')
        self.messages = OutputListView()
        self.messages.setFont(font)
        self.messages.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.model = OutputModel(self.messages.fontMetrics())
        self.filter = OutputFilter(self, self.model)
        self.filter.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.filter.setDynamicSortFilter(True)
        self.messages.setModel(self.filter)
        self.messages.clicked.connect(self.line_clicked)

        toolbar = QToolBar()
        set_icon_size(toolbar)

        self.action_error = QAction(self)
        self.action_error.setText("&Error")
        self.action_error.setCheckable(True)
        self.action_error.setChecked(launcher.error.isChecked())
        self.action_error.triggered.connect(self.update_filter)
        self.action_error.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        CONFIG.set_shortcut(self.action_error, 'launcher_show_error')
        self.action_warning = QAction(self)
        self.action_warning.setText("&Warning")
        self.action_warning.setCheckable(True)
        self.action_warning.setChecked(launcher.warning.isChecked())
        self.action_warning.triggered.connect(self.update_filter)
        self.action_warning.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        CONFIG.set_shortcut(self.action_warning, 'launcher_show_warning')
        self.action_important = QAction(self)
        self.action_important.setText("I&mportant")
        self.action_important.setCheckable(True)
        self.action_important.setChecked(launcher.important.isChecked())
        self.action_important.triggered.connect(self.update_filter)
        self.action_important.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        CONFIG.set_shortcut(self.action_important, 'launcher_show_important')
        self.action_info = QAction(self)
        self.action_info.setText("&Info")
        self.action_info.setCheckable(True)
        self.action_info.setChecked(launcher.info.isChecked())
        self.action_info.triggered.connect(self.update_filter)
        self.action_info.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        CONFIG.set_shortcut(self.action_info, 'launcher_show_info')
        self.action_result = QAction(self)
        self.action_result.setText("&Result")
        self.action_result.setCheckable(True)
        self.action_result.setChecked(launcher.result.isChecked())
        self.action_result.triggered.connect(self.update_filter)
        self.action_result.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        CONFIG.set_shortcut(self.action_result, 'launcher_show_result')
        self.action_data = QAction(self)
        self.action_data.setText("&Data")
        self.action_data.setCheckable(True)
        self.action_data.setChecked(launcher.data.isChecked())
        self.action_data.triggered.connect(self.update_filter)
        self.action_data.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        CONFIG.set_shortcut(self.action_data, 'launcher_show_data')
        self.action_detail = QAction(self)
        self.action_detail.setText("De&tail")
        self.action_detail.setCheckable(True)
        self.action_detail.setChecked(launcher.detail.isChecked())
        self.action_detail.triggered.connect(self.update_filter)
        self.action_detail.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        CONFIG.set_shortcut(self.action_detail, 'launcher_show_detail')
        self.action_debug = QAction(self)
        self.action_debug.setText("De&bug")
        self.action_debug.setCheckable(True)
        self.action_debug.setChecked(launcher.debug.isChecked())
        self.action_debug.triggered.connect(self.update_filter)
        self.action_debug.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        CONFIG.set_shortcut(self.action_debug, 'launcher_show_debug')

        self.addAction(self.action_error)
        self.addAction(self.action_warning)
        self.addAction(self.action_important)
        self.addAction(self.action_info)
        self.addAction(self.action_result)
        self.addAction(self.action_data)
        self.addAction(self.action_detail)
        self.addAction(self.action_debug)

        self.levels = (
            self.action_error,
            self.action_error,
            self.action_warning,
            self.action_important,
            self.action_info,
            self.action_result,
            self.action_data,
            self.action_detail,
            self.action_error,
            self.action_debug
        )

        view_menu = QMenu("Show", self)
        view_menu.addAction(self.action_error)
        view_menu.addAction(self.action_warning)
        view_menu.addAction(self.action_important)
        view_menu.addAction(self.action_info)
        view_menu.addAction(self.action_result)
        view_menu.addAction(self.action_data)
        view_menu.addAction(self.action_detail)
        view_menu.addAction(self.action_debug)

        menu_button = QToolButton()
        menu_button.setMenu(view_menu)
        menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_button.setIcon(QIcon.fromTheme('edit-find'))
        menu_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # menu_button.setText("Log Levels")
        tool_action = QWidgetAction(self)
        tool_action.setDefaultWidget(menu_button)
        toolbar.addAction(tool_action)

        self.halt_action = QAction(QIcon.fromTheme('process-stop'), "Halt", self)
        CONFIG.set_shortcut(self.halt_action, 'launcher_halt')
        self.halt_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.halt_action.triggered.connect(self.halt_thread)
        self.messages.addAction(self.halt_action)
        toolbar.addAction(self.halt_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar.addWidget(spacer)

        label = QLabel("&Filter: ")
        toolbar.addWidget(label)

        self.search = LineEditWithClear()
        self.search.setMaximumWidth(400)
        label.setBuddy(self.search)
        toolbar.addWidget(self.search)
        self.search.textChanged.connect(self.filter_text)

        close_all_action = QAction(QIcon.fromTheme('window-close'), "Clea&nup All", self)
        CONFIG.set_shortcut(close_all_action, 'launcher_cleanup')
        close_all_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        close_all_action.setToolTip("Close all docks with finished processes")
        close_all_action.triggered.connect(self.close_all_stopped_docks)
        self.messages.addAction(close_all_action)
        toolbar.addAction(close_all_action)

        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self.messages)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setWidget(widget)

        close_action = QAction("Clos&e", self)
        CONFIG.set_shortcut(close_action, 'launcher_close')
        close_action.triggered.connect(self.close)
        close_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.addAction(close_action)

        before_action = self.messages.context_menu.actions()[0]
        self.messages.context_menu.insertAction(before_action, self.action_error)
        self.messages.context_menu.insertAction(before_action, self.action_warning)
        self.messages.context_menu.insertAction(before_action, self.action_important)
        self.messages.context_menu.insertAction(before_action, self.action_info)
        self.messages.context_menu.insertAction(before_action, self.action_result)
        self.messages.context_menu.insertAction(before_action, self.action_data)
        self.messages.context_menu.insertAction(before_action, self.action_detail)
        self.messages.context_menu.insertAction(before_action, self.action_debug)
        self.messages.context_menu.insertSeparator(before_action)
        self.messages.context_menu.insertAction(before_action, self.halt_action)
        self.messages.context_menu.insertAction(before_action, close_action)
        self.messages.context_menu.insertAction(before_action, close_all_action)
        self.messages.context_menu.insertSeparator(before_action)

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

        self.mutex = QMutex()
        self.thread = None

        self.newlines = []

        if filename is not None:
            fd, fb = (s.replace(' ', '&nbsp;') for s in os.path.split(filename))
            sep = os.path.sep
            if sep == '\\':
                sep = '\\\\'
                fd = fd.replace('\\', '\\\\')
            self.link = re.compile(
                r'((?:{}{})?{}(?:(?:|,|:| in <\S+>,?)(?: XML)? line |:))(\d+)(.*)'.format(fd, sep, fb))
        else:
            self.link = None

    def reconfig(self):
        font = self.messages.font()
        if set_font(font, 'launcher_local/font'):
            self.messages.setFont(font)
        self.model.update_font(self.messages.fontMetrics())
        self.filter.invalidate()

    def line_clicked(self, index):
        line = self.filter.data(index, LINE_ROLE)
        if line is not None:
            parent = self.parent()
            if parent:
                parent.goto_line(line)

    def update_filter(self):
        self.filter.invalidateFilter()

    def filter_text(self, text):
        self.filter.setFilterFixedString(text)

    def parse_line(self, line, fname=None):
        if not line: return
        try:
            line = line.decode(self.main_window.document.coding)
        except UnicodeDecodeError:
            try:
                line = line.decode('utf-8')
            except UnicodeDecodeError:
                line = line.decode('cp1250')
        level = {'CRITICAL ERROR:': LEVEL_CRITICAL_ERROR,
                 'ERROR         :': LEVEL_ERROR,
                 'WARNING       :': LEVEL_WARNING,
                 'IMPORTANT     :': LEVEL_IMPORTANT,
                 'INFO          :': LEVEL_INFO,
                 'RESULT        :': LEVEL_RESULT,
                 'DATA          :': LEVEL_DATA,
                 'DETAIL        :': LEVEL_DETAIL,
                 'ERROR DETAIL  :': LEVEL_ERROR_DETAIL,
                 'DEBUG         :': LEVEL_DEBUG}.get(line[:15], 0)
        lineno = None
        if self.link is not None:
            match = self.link.search(line)
            if match is not None:
                lineno = int(match.groups()[1])
        try:
            self.mutex.lock()
            self.newlines.append((level, line, lineno))
        finally:
            self.mutex.unlock()

    def update_output(self):
        try:
            self.mutex.lock()
            for data in self.newlines:
                self.model.add_line(*data)
            self.newlines = []
        finally:
            self.mutex.unlock()
        #self.filter.invalidateFilter()
        move = self.messages.verticalScrollBar().value() == self.messages.verticalScrollBar().maximum()
        if move:
            self.messages.scrollToBottom()

    def halt_thread(self):
        confirm = QMessageBox.question(self, "Halt Process",
                                             "PLaSK is currently running. Do you really want to terminate it? "
                                             "All computation results may be lost!",
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes and self.thread is not None:
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
                                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.Yes:
                self.thread.kill_process()
                if not self.thread.wait(6000):
                    QMessageBox.critical(self, "Close Window",
                                               "PLaSK process could not be terminated. Window will not be closed. "
                                               "Please try once again or contact the program authors.",
                                               QMessageBox.StandardButton.Ok)
                    event.ignore()
                    event.ignore()
            else:
                event.ignore()

    def closeEvent(self, event):
        focus = self.messages.hasFocus()
        self.check_close_event(event)
        if not event.isAccepted():
            return
        super().closeEvent(event)
        if focus:
            main_window = self.parent()
            others = [w for w in main_window.findChildren(QDockWidget)
                      if isinstance(w, OutputWindow) and w is not self and w.isVisible()]
            if others:
                others[-1].messages.setFocus()

    def close_all_stopped_docks(self):
        main_window = self.parent()
        docks = (w for w in main_window.findChildren(QDockWidget)
                 if isinstance(w, OutputWindow) and w.isVisible() and
                 not (w.thread is None or w.thread.isRunning()))
        for dock in docks:
            dock.close()
