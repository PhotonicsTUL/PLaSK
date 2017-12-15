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

from ..utils.widgets import LineEditWithClear, set_icon_size

from ..qt.QtCore import *
from ..qt.QtGui import *
from ..qt.QtWidgets import *

from ..utils.config import CONFIG


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

LEVEL_ROLE = Qt.UserRole
LINE_ROLE = Qt.UserRole + 1

class OutputModel(QAbstractListModel):

    def __init__(self, fm):
        super(OutputModel, self).__init__()
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
        if role == Qt.DisplayRole:
            return self.lines[row][1]
        if role == Qt.ForegroundRole:
            level = self.lines[row][0]
            color = [
                'black',    # default
                'red',      # critical error
                'red',      # error
                'brown',    # warning
                'magenta',  # important
                'blue',     # info
                'green',    # result
                '#006060',  # data
                '#404040',  # detail
                '#800000',  # error detail
                'gray',     # debug
            ][level]
            return QBrush(QColor(color))
        if role == LEVEL_ROLE:
            return self.lines[row][0]
        if role == LINE_ROLE:
            return self.lines[row][2]
        if role == Qt.SizeHintRole and self.fm is not None:
            return QSize(self.fm.width(self.lines[row][1])+self.lw, self.lh)

    def rowCount(self, parent=None):
        return len(self.lines)

    # def columnCount(self, parent=None):
    #     return 2


class OutputListView(QListView):

    def __init__(self, parent=None):
        super(OutputListView, self).__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        super(OutputListView, self).mouseMoveEvent(event)
        index = self.indexAt(event.pos())
        line = self.model().data(index, LINE_ROLE)
        if line is not None:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event):
        if event == QKeySequence.Copy:
            self.copy()
            event.accept()
        else:
            super(OutputListView, self).keyPressEvent(event)

    def copy(self):
        rows = self.selectionModel().selectedRows()
        rows.sort(key=lambda row: row.row())
        lines = [self.model().data(row, Qt.DisplayRole) for row in rows]
        QApplication.clipboard().setText('\n'.join(lines))


class OutputFilter(QSortFilterProxyModel):

    def __init__(self, window, model):
        super(OutputFilter, self).__init__()
        self.window = window
        self.setSourceModel(model)

    def filterAcceptsRow(self, row, parent):
        try:
            level = self.sourceModel().lines[row][0]
        except IndexError:
            return False
        if level != 0 and not self.window.levels[level-1].isChecked():
            return False
        return super(OutputFilter, self).filterAcceptsRow(row, parent)


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
        self.messages = OutputListView()
        self.messages.setFont(font)
        self.messages.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.model = OutputModel(self.messages.fontMetrics())
        self.filter = OutputFilter(self, self.model)
        self.filter.setFilterCaseSensitivity(Qt.CaseInsensitive)
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
        self.action_error.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_error.setShortcut('1')
        self.action_warning = QAction(self)
        self.action_warning.setText("&Warning")
        self.action_warning.setCheckable(True)
        self.action_warning.setChecked(launcher.warning.isChecked())
        self.action_warning.triggered.connect(self.update_filter)
        self.action_warning.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_warning.setShortcut('2')
        self.action_important = QAction(self)
        self.action_important.setText("I&mportant")
        self.action_important.setCheckable(True)
        self.action_important.setChecked(launcher.important.isChecked())
        self.action_important.triggered.connect(self.update_filter)
        self.action_important.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_important.setShortcut('3')
        self.action_info = QAction(self)
        self.action_info.setText("&Info")
        self.action_info.setCheckable(True)
        self.action_info.setChecked(launcher.info.isChecked())
        self.action_info.triggered.connect(self.update_filter)
        self.action_info.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_info.setShortcut('4')
        self.action_result = QAction(self)
        self.action_result.setText("&Result")
        self.action_result.setCheckable(True)
        self.action_result.setChecked(launcher.result.isChecked())
        self.action_result.triggered.connect(self.update_filter)
        self.action_result.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_result.setShortcut('5')
        self.action_data = QAction(self)
        self.action_data.setText("&Data")
        self.action_data.setCheckable(True)
        self.action_data.setChecked(launcher.data.isChecked())
        self.action_data.triggered.connect(self.update_filter)
        self.action_data.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_data.setShortcut('6')
        self.action_detail = QAction(self)
        self.action_detail.setText("De&tail")
        self.action_detail.setCheckable(True)
        self.action_detail.setChecked(launcher.detail.isChecked())
        self.action_detail.triggered.connect(self.update_filter)
        self.action_detail.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_detail.setShortcut('7')
        self.action_debug = QAction(self)
        self.action_debug.setText("De&bug")
        self.action_debug.setCheckable(True)
        self.action_debug.setChecked(launcher.debug.isChecked())
        self.action_debug.triggered.connect(self.update_filter)
        self.action_debug.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_debug.setShortcut('8')

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

        view_menu = QMenu("Show")
        view_menu.addAction(self.action_error)
        view_menu.addAction(self.action_warning)
        view_menu.addAction(self.action_important)
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

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        toolbar.addWidget(spacer)

        label = QLabel("&Filter: ")
        toolbar.addWidget(label)

        self.search = LineEditWithClear()
        self.search.setMaximumWidth(400)
        label.setBuddy(self.search)
        toolbar.addWidget(self.search)
        self.search.textChanged.connect(self.filter_text)

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

        self.mutex = QMutex()
        self.thread = None

        self.newlines = []

    def reconfig(self):
        font = self.messages.font()
        if font.fromString(','.join(CONFIG['launcher_local/font'])):
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

    def parse_line(self, line, link=None):
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
        if link is not None:
            match = link.search(line)
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
