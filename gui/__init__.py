#!/usr/bin/env python3
# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2023 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import sys
import os
import re
import subprocess
import pkgutil
import traceback
from datetime import date

from lxml import etree
from uuid import getnode

try:
    import plask
except ImportError:
    pass

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import gui
    gui.main()

ACTIONS = []
HELP_ACTIONS = []

basestring = str, bytes
try:
    _DEBUG = bool(eval(os.environ['PLASKGUI_DEBUG'].title()))
except (KeyError, ValueError):
    _DEBUG = False

from .qt.QtCore import *
from .qt.QtWidgets import *
from .qt.QtGui import *
from .qt import QtSignal, QT_API, qt_exec


sys.path.insert(4, os.path.join(__path__[0], 'external', 'pysparkle'))


# Set-up correct backend for matplotlib
try:
    import matplotlib
except ImportError:
    matplotlib = None

from .utils.config import CONFIG

from .xpldocument import XPLDocument
from .textdocument import TextDocument, PyDocument, XmlDocument
from .model.info import Info
from .launch import launch_plask
from .controller.materials.plot import show_material_plot
from .controller.multi import GUIAndSourceController

from .utils.config import ConfigProxy, dark_style
from .utils.settings import SettingsDialog
from .utils.texteditor import update_textedit
from .utils.texteditor.python.completer import prepare_completions
from .utils.widgets import fire_edit_end
from .utils.help import open_help


try:
    from pysparkle import PySparkle
except:
    PySparkle = None


WINDOWS = set()


CURRENT_DIR = CONFIG['session/recent_dir']
if CURRENT_DIR is None:
    CURRENT_DIR = os.getcwd()


def load_recent_files():
    CONFIG.sync()
    recent = CONFIG['session/recent_files']
    if recent is None:
        recent = []
    elif type(recent) is not list:
        recent = [recent]
    return recent[-10:]


def update_recent_files(filename):
    global CURRENT_DIR
    filename = filename.replace('/', os.path.sep)
    CURRENT_DIR = os.path.dirname(filename)
    CONFIG['session/recent_dir'] = CURRENT_DIR
    recent = load_recent_files()
    try:
        recent.remove(filename)
    except ValueError:
        pass
    if os.path.isfile(filename):
        recent = recent[-9:]
        recent.append(filename)
    CONFIG['session/recent_files'] = recent
    CONFIG.sync()


def close_all_windows():
    for window in WINDOWS.copy():
        if not window.close():
            return False
    return True


if matplotlib is not None:
    matplotlib.rcParams['axes.facecolor'] = CONFIG['plots/face_color']
    matplotlib.rcParams['axes.edgecolor'] = CONFIG['plots/edge_color']
    matplotlib.rcParams['grid.color'] = CONFIG['plots/grid_color']


class MainMenuAction(QAction):

    def __init__(self, button, parent=None):
        super().__init__(parent)
        self.button = button

    def toolTip(self):
        return self.button.toolTip()

    def setToolTip(self, tooltip):
        self.button.setToolTip(tooltip)


class MainWindow(QMainWindow):

    SECTION_TITLES = dict(defines=" &Defines ", materials=" &Materials ", geometry=" &Geometry ", grids=" M&eshing ",
                          solvers=" &Solvers ", connects=" &Connects ", script=" Sc&ript ", source=" Source ")

    # icons: http://standards.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
    SECTION_ICONS = {
        'defines': 'accessories-dictionary',
        'materials': 'accessories-character-map',
        'geometry': 'system-file-manager',
        'grids': 'preferences-desktop-keyboard',
        'solvers': 'utilities-system-monitor',
        'connects': 'preferences-desktop-accessibility',
        'script': 'accessories-text-editor',
        'source': 'accessories-text-editor',
    }

    SECTION_TIPS = {
        'defines': "Edit the list of pre-defined variables for use in the rest of the file  (Alt+D)",
        'materials': "Edit custom materials  (Alt+M)",
        'geometry': "Edit geometries of your structures  (Alt+G)",
        'grids': "Edit computational meshes or set-up automatic mesh generators  (Alt+E)",
        'solvers': "Create and configure computational solvers  (Alt+S)",
        'connects': "Define connections between computational solvers  (Alt+C)",
        'script': "Edit control script for your computations (Alt+R)",
        'source': "XPL source code — this view has been opened because your file is not a correct XML "
                  "and cannot be loaded in the GUI editor",
    }

    shown = QtSignal()
    closing = QtSignal(QCloseEvent)
    closed = QtSignal()
    config_changed = QtSignal()

    def __init__(self, filename=None, Document=XPLDocument):
        super().__init__()

        self.document = None

        self.current_tab_index = -1
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged[int].connect(self.tab_change)
        # self.tabs.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        menu_bar = QMenuBar(self)
        menu_bar.setVisible(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs)
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.showsource_action = QAction(
            QIcon.fromTheme('show-source'),
            'Toggle sour&ce view', self)
        CONFIG.set_shortcut(self.showsource_action, 'show_source')
        self.showsource_action.setCheckable(True)
        self.showsource_action.setStatusTip('Show XPL source of the current section')
        self.showsource_action.setEnabled(False)

        self.setWindowIcon(QIcon.fromTheme('plaskgui'))

        if filename is None or not self._try_load_from_file(filename):  # try to load only if filename is not None
            self.document = Document(self)
            self.setup_model()
            self.setWindowTitle("[*] PLaSK")
            self.setWindowModified(False)

        new_action = QAction(QIcon.fromTheme('document-new'), '&New', self)
        CONFIG.set_shortcut(new_action, 'new_xpl')
        new_action.setStatusTip('Create a new XPL file')
        new_action.triggered.connect(lambda: self.new(XPLDocument))

        newpy_action = QAction(QIcon.fromTheme('document-new'), 'New &Python', self)
        CONFIG.set_shortcut(newpy_action, 'new_python')
        newpy_action.setStatusTip('Create a new Python file')
        newpy_action.triggered.connect(lambda: self.new(PyDocument))

        open_action = QAction(QIcon.fromTheme('document-open'), '&Open...', self)
        CONFIG.set_shortcut(open_action, 'open_file')
        open_action.setStatusTip('Open an existing data file')
        open_action.triggered.connect(self.open)

        save_action = QAction(QIcon.fromTheme('document-save'), '&Save', self)
        CONFIG.set_shortcut(save_action, 'save_file')
        save_action.setStatusTip('Save the file to disk')
        save_action.triggered.connect(self.save)

        saveas_action = QAction(QIcon.fromTheme('document-save-as'), 'Save &As...', self)
        CONFIG.set_shortcut(saveas_action, 'saveas_file')
        saveas_action.setStatusTip('Save the file to disk asking for a new name')
        saveas_action.triggered.connect(self.save_as)

        reload_action = QAction(QIcon.fromTheme('view-refresh'), '&Reload', self)
        CONFIG.set_shortcut(reload_action, 'reload_file')
        reload_action.setStatusTip('Reload the current file from disk')
        reload_action.triggered.connect(self.reload)

        launch_action = QAction(QIcon.fromTheme('media-playback-start'), '&Launch...', self)
        CONFIG.set_shortcut(launch_action, 'launch')
        launch_action.setStatusTip('Launch the current file in PLaSK')
        launch_action.triggered.connect(lambda: launch_plask(self))

        goto_action = QAction(QIcon.fromTheme('go-jump'), '&Go to Line...', self)
        CONFIG.set_shortcut(goto_action, 'goto_line')
        goto_action.setStatusTip('Go to the specified line')
        goto_action.triggered.connect(self.on_goto_line)

        plot_material_action = QAction(QIcon.fromTheme('matplotlib'), 'Examine &Material Parameters...', self)
        CONFIG.set_shortcut(plot_material_action, 'examine_material')
        plot_material_action.triggered.connect(lambda: show_material_plot(self, self.document.materials.model,
                                                                          getattr(self.document.defines, 'model', None)))

        fullscreen_action = QAction(QIcon.fromTheme('view-fullscreen'), 'Toggle Full Screen', self)
        CONFIG.set_shortcut(fullscreen_action, 'fullscreen')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)

        settings_action = QAction(QIcon.fromTheme('document-properties'), 'GUI Se&ttings...', self)
        CONFIG.set_shortcut(settings_action, 'settings')
        settings_action.setStatusTip('Change some GUI settings')
        settings_action.triggered.connect(self.show_settings)

        about_action = QAction(QIcon.fromTheme('dialog-information'), 'A&bout...', self)
        CONFIG.set_shortcut(about_action, 'about')
        about_action.setStatusTip('Show information about PLaSK')
        about_action.triggered.connect(self.about)

        help_action = QAction(QIcon.fromTheme('help-contents'), 'Open &Help...', self)
        CONFIG.set_shortcut(help_action, 'help')
        help_action.setStatusTip('Open on-line help')
        help_action.triggered.connect(lambda: open_help(main_window=self))

        exit_action = QAction(QIcon.fromTheme('application-exit'), 'E&xit', self)
        CONFIG.set_shortcut(exit_action, 'quit')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)

        self.recent_menu = QMenu('Open R&ecent')
        self.recent_menu.setIcon(QIcon.fromTheme('document-open-recent'))
        self.recent_menu.aboutToShow.connect(self.update_recent_menu)

        self.menu = QMenu('&PLaSK')

        self.menu.addAction(new_action)
        self.menu.addAction(newpy_action)
        self.menu.addAction(open_action)
        self.menu.addMenu(self.recent_menu)
        self.menu.addAction(save_action)
        self.menu.addAction(saveas_action)
        self.menu.addAction(reload_action)
        self.menu.addSeparator()
        self.menu.addAction(goto_action)
        self.menu.addAction(self.showsource_action)
        self.menu.addAction(launch_action)
        self.menu.addSeparator()
        self.menu.addAction(plot_material_action)
        if ACTIONS:
            self.menu.addSeparator()
            for op in ACTIONS:   # for plugins use
                if op is not None:
                    self.menu.addAction(op(self))
                else:
                    self.menu.addSeparator()
        self.menu.addSeparator()
        self.menu.addAction(about_action)
        self.menu.addAction(help_action)
        for op in HELP_ACTIONS:   # for plugins use
            if op is not None:
                self.menu.addAction(op(self))
            else:
                self.menu.addSeparator()
        self._pysparkle_place = self.menu.addSeparator()
        self.menu.addAction(fullscreen_action)
        self.menu.addAction(settings_action)
        self.menu.addSeparator()
        self.menu.addAction(exit_action)

        if os.name == 'nt':
            menu_button = QToolButton(self)
            menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            menu_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            menu_button.setAutoFillBackground(True)
            font = menu_button.font()
            font.setBold(True)
            menu_button.setFont(font)
        else:
            menu_button = QPushButton(self)
        menu_button.setText("PLaSK")
        pal = menu_button.palette()
        if CONFIG['main_window/dark_style']:
            pal.setColor(QPalette.ColorRole.Button, QColor(42, 130, 218))
        else:
            pal.setColor(QPalette.ColorRole.Button, QColor("#88aaff"))
        menu_button.setIcon(QIcon.fromTheme('plask-logo'))
        menu_button.setPalette(pal)
        menu_button.setToolTip("Show main menu")
        menu_action = MainMenuAction(menu_button)
        menu_action.triggered.connect(menu_button.showMenu)
        CONFIG.set_shortcut(menu_action, 'main_menu')
        self.addAction(menu_action)

        menu_button.setMenu(self.menu)
        self.tabs.setCornerWidget(menu_button, Qt.Corner.TopLeftCorner)

        tabs_menu = QMenu("Sections", menu_bar)
        def add_tab_menu(indx):
            def show_tab():
                self.tabs.setCurrentIndex(indx)
                self.tab_change(indx)
            tabs_action.triggered.connect(show_tab)
        for i in range(self.tabs.count()):
            tabs_action = QAction(self.tabs.tabText(i), self)
            add_tab_menu(i)
            tabs_menu.addAction(tabs_action)
        menu_bar.addMenu(self.menu)
        menu_bar.addMenu(tabs_menu)

        source_button = QToolButton(self)
        source_button.setDefaultAction(self.showsource_action)
        self.tabs.setCornerWidget(source_button, Qt.Corner.TopRightCorner)

        self.shown.connect(self.init_pysparkle, Qt.ConnectionType.QueuedConnection)

        fs = int(1.3 * QFont().pointSize())
        self.tabs.setStyleSheet("QTabBar {{ font-size: {}pt; }}".format(fs))
        menu_button.setStyleSheet("QPushButton {{ font-size: {}pt; font-weight: bold; }}".format(fs))

        self.config_changed.connect(update_textedit)

        geometry = CONFIG['session/geometry']
        if geometry is not None:
            try:
                screen = QApplication.screenAt(geometry.center()).availableGeometry()
            except AttributeError:
                screen = QDesktopWidget().availableGeometry(geometry.center())
            geometry.setWidth(min(geometry.width(), screen.right()-geometry.left()+1))
            geometry.setHeight(min(geometry.height(), screen.bottom()-geometry.top()+1))
            self.setGeometry(geometry)
        else:
            try:
                screen = self.screen().availableGeometry()
            except AttributeError:
                screen = QDesktopWidget().availableGeometry(self)
            self.resize(int(screen.width() * 0.8), int(screen.height() * 0.9))

        self.setAcceptDrops(True)


        state = Qt.WindowState.WindowNoState
        if CONFIG['session/maximized']: state |= Qt.WindowState.WindowMaximized
        if CONFIG['session/fullscreen']: state |= Qt.WindowState.WindowFullScreen
        self.setWindowState(state)

        self.show()

    def dragEnterEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            for url in mime_data.urls():
                if url.isLocalFile():
                    event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                if self.load_file(url.toLocalFile()):
                    event.acceptProposedAction()


    def showEvent(self, event):
        super().showEvent(event)
        self.shown.emit()

    @property
    def current_controller(self):
        if self.current_tab_index == -1: return None
        return self.document.controller_by_index(self.current_tab_index)

    def update_recent_menu(self):
        self.recent_menu.clear()
        class Func:
            def __init__(s, f): s.f = f
            def __call__(s): return self.open(s.f)
        for i,f in enumerate(reversed(load_recent_files())):
            action = QAction(f, self)
            action.triggered.connect(Func(f))
            if i < 9:
                action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key(int(Qt.Key.Key_1) + i)))
            elif i == 9:
                action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_0))
            self.recent_menu.addAction(action)

    def _try_load_from_file(self, filename, tab=None):
        update_recent_files(os.path.abspath(filename))
        document = PyDocument(self) if filename.endswith('.py') else XPLDocument(self)
        try:
            try:
                document.load_from_file(filename)
            except (etree.LxmlError, ValueError) as e:
                if _DEBUG:
                    traceback.print_exc()
                document = XmlDocument(self)
                document.load_from_file(filename)
                QMessageBox.warning(self, "XML parse error",
                                        "Error while parsing file '{}':\n{}\n\n"
                                        "Opening in text mode!".format(filename, str(e)))
        except Exception as e:
            if _DEBUG: raise e
            QMessageBox.critical(self, "File load error",
                                       "Error while loading file '{}':\n{}".format(filename, str(e)))
            return False

        self.document = document
        self.setup_model(tab)
        self.set_changed(False)
        os.chdir(os.path.dirname(os.path.abspath(filename)))
        return True

    def setup_model(self, tab=None):
        self.tabs.clear()
        for m in self.document.SECTION_NAMES:
            self.tabs.addTab(self.document.controller_by_name(m).get_widget(), self.SECTION_TITLES[m])
            self.tabs.setTabToolTip(self.tabs.count()-1, self.SECTION_TIPS[m])
            # self.tabs.setTabIcon(self.tabs.count()-1, QIcon.fromTheme(SECTION_ICONS[m]))
        self.current_tab_index = -1
        if isinstance(self.document, TextDocument):
            self.tab_change(0)
        else:
            if tab is None: tab = 2
            self.tabs.setCurrentIndex(tab)

    def new(self, Document=XPLDocument):
        new_window = MainWindow(Document=Document)
        new_window.resize(self.size())
        new_window.move(self.x() + 24, self.y() + 24)
        WINDOWS.add(new_window)

    def open(self, filename=None):
        if not filename:
            filename = QFileDialog.getOpenFileName(self, "Open File", CURRENT_DIR,
                                                         "PLaSK file (*.xpl *.py);;"
                                                         "PLaSK structure data (*.xpl);;"
                                                         "Python script (*.py)")
            if type(filename) == tuple: filename = filename[0]
            if not filename:
                return
        self.load_file(filename)

    def load_file(self, filename):
        if self.document.filename is None and not self.isWindowModified():
            self.document.controller_by_index(self.current_tab_index).on_edit_exit()
            self.current_tab_index = -1
            self._try_load_from_file(filename)
            self.current_section_enter()
            new_window = self
        else:
            new_window = MainWindow(filename)
            try:
                if new_window.document.filename is not None:
                    new_window.resize(self.size())
                    WINDOWS.add(new_window)
                    new_window.move(self.x() + 24, self.y() + 24)
                else:
                    new_window.setWindowModified(False)
                    new_window.close()
            except AttributeError:
                new_window.setWindowModified(False)
                new_window.close()
                new_window = None
            else:
                new_window.raise_()
        return new_window

    def reload(self):
        if self.document.filename is None:
            return
        if self.isWindowModified():
            confirm = QMessageBox.question(self, "Unsaved Changes",
                                           "File has unsaved changes. Do you want to discard them and reload the file?",
                                           QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,  QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.No:
                return
        current_tab_index = self.current_tab_index
        self.document.controller_by_index(current_tab_index).on_edit_exit()
        self.current_tab_index = -1
        if not self._try_load_from_file(self.document.filename, current_tab_index):
            self.setWindowModified(True)

    def _save_document(self, filename):
        fire_edit_end()
        try:
            self.document.save_to_file(str(filename))
        except Exception as err:
            if _DEBUG:
                traceback.print_exc()
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Save Error")
            msgbox.setText("The file '{}' could not be saved to disk.".format(filename))
            msgbox.setInformativeText(str(err))
            msgbox.setStandardButtons(QMessageBox.StandardButton.Ok)
            msgbox.setIcon(QMessageBox.Icon.Critical)
            qt_exec(msgbox)
            return False
        else:
            abspath = os.path.abspath(filename)
            update_recent_files(abspath)
            self.setWindowModified(False)
            self.setWindowTitle(u"{}[*] - PLaSK".format(self.document.filename))
            os.chdir(os.path.dirname(abspath))
            return True

    def save(self):
        if self.document.filename is not None:
            if not self.before_save():
                return False
            return self._save_document(self.document.filename)
        else:
            return self.save_as()

    def save_as(self):
        """Ask for filename and save to c
                msgbox.setDefaultButton(QMessageBox.StandardButton.Yes);hosen file. Return true only when file has been saved."""
        if not self.before_save():
            return False
        flt = "{} (*.{})".format(self.document.NAME, self.document.EXT)
        filename = QFileDialog.getSaveFileName(self, "Save file as", self.document.filename or CURRENT_DIR, flt)
        if type(filename) is tuple:
            filename = filename[0]
        if not filename:
            return False
        if self.document.filename is None and not filename.endswith('.'+self.document.EXT):
                filename += '.' + self.document.EXT
        return self._save_document(filename)

    def before_save(self):
        """"Is called just before save, return True if document can be saved."""
        try:
            QApplication.focusWidget().editingFinished.emit()
        except AttributeError:
            pass

        if self.current_tab_index != -1:
            try:
                self.document.controller_by_index(self.current_tab_index).save_data_in_model()
            except (etree.LxmlError, ValueError):
                pass  # error is set in the controller

        for contrl in self.document.controllers:
            if not contrl.can_save():
                return False

        errors = self.document.get_info(Info.ERROR)
        if errors:
            msgbox = QMessageBox()
            msgbox.setText("Document contains some non-critical errors.\n\n"
                           "It is possible to save it, however launching it will most probably fail.")
            msgbox.setDetailedText('\n'.join(map(str, errors)))
            msgbox.setInformativeText("Do you want to save anyway?")
            msgbox.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msgbox.setIcon(QMessageBox.Icon.Warning)
            msgbox.setDefaultButton(QMessageBox.StandardButton.Yes)
            return qt_exec(msgbox) == QMessageBox.StandardButton.Yes
        return True

    def show_settings(self):
        dialog = SettingsDialog(self)
        qt_exec(dialog)

    def current_section_exit(self):
        """"Should be called just before leaving the current section."""
        if self.current_tab_index != -1:
            if not self.document.controller_by_index(self.current_tab_index).on_edit_exit():
                self.tabs.setCurrentIndex(self.current_tab_index)
                return False
        return True

    def current_section_enter(self):
        """"Should be called just after setting the current section."""
        if self.current_tab_index != -1:
            c = self.document.controller_by_index(self.current_tab_index)
            c.on_edit_enter()

    def tab_change(self, index):
        if index == self.current_tab_index: return
        if not self.current_section_exit():
            self.tabs.setCurrentIndex(self.current_tab_index)
            return
        self.current_tab_index = index
        self.current_section_enter()

    def remove_tools_actions(self, actions):
        for action in actions:
            self.tools_menu.removeAction(action)

    def closeEvent(self, event):

        try:
            QApplication.focusWidget().editingFinished.emit()
        except AttributeError:
            pass

        self.closing.emit(event)
        if not event.isAccepted():
            return

        if self.isWindowModified():
            confirm = QMessageBox.question(self, "Unsaved File",
                                                 "File is not saved. Do you want to save it before closing the window?",
                                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if confirm == QMessageBox.StandardButton.Cancel or (confirm == QMessageBox.StandardButton.Yes and not self.save()):
                event.ignore()
                return

        self.closed.emit()
        self.document = None

        try:
            WINDOWS.remove(self)
        except KeyError:
            pass

        fullscreen = self.isFullScreen()
        maximized = self.isMaximized()
        CONFIG['session/fullscreen'] = fullscreen
        CONFIG['session/maximized'] = maximized
        if not (fullscreen or maximized):
            geometry = self.geometry()
            CONFIG['session/geometry'] = geometry
        CONFIG.sync()

    def set_changed(self, changed):
        """Set changed flags in the document window"""
        try:
            if self.document and self.document.filename:
                self.setWindowTitle(u"{}[*] - PLaSK".format(self.document.filename))
            else:
                self.setWindowTitle("[*] PLaSK")
        except AttributeError:
            self.setWindowTitle("[*] PLaSK")
        self.setWindowModified(changed)

    def on_goto_line(self):
        dialog = GotoDialog(self)
        if qt_exec(dialog):
            self.goto_line(int(dialog.input.text()))
        else:
            return

    def goto_line(self, line_number=None):
        if line_number is None:
            dialog = GotoDialog(self)
            if qt_exec(dialog):
                line_number = int(dialog.input.text())
            else:
                return
        indx = None
        for i, c in enumerate(self.document.controllers):
            if c.model.line_in_file is None: continue
            if line_number < c.model.line_in_file: break
            indx = i
            cntrl = c
            lineno = line_number - c.model.line_in_file - 1
        if indx is not None:
            self.tabs.setCurrentIndex(indx)
            self.tab_change(indx)
            if self.showsource_action.isEnabled() and not self.showsource_action.isChecked():
                self.showsource_action.trigger()
            editor = cntrl.get_source_widget().editor
            cursor = QTextCursor(editor.document().findBlockByLineNumber(
                min(lineno, editor.document().blockCount()-1)))
            editor.setTextCursor(cursor)
            editor.setFocus()

    def set_show_source_state(self, show_source_enabled):
        if show_source_enabled is None:
            self.showsource_action.setEnabled(False)
        else:
            self.showsource_action.setEnabled(True)
            self.showsource_action.setChecked(show_source_enabled)

    def get_show_source_state(self, do_enabled=False):
        if do_enabled:
            self.showsource_action.setEnabled(True)
        else:
            if not self.showsource_action.isEnabled(): return None
        return self.showsource_action.isChecked()

    def init_pysparkle(self):
        global pysparkle
        if pysparkle is None and PySparkle is not None:
            if VERSION is not None:
                pysparkle = PySparkle("https://plask.app/appcast.xml", "PLaSK", VERSION[:10],
                                      config=ConfigProxy('updates'), shutdown=close_all_windows,
                                      frontend='qt')
        if pysparkle is not None:
            action_check_update = QAction(QIcon.fromTheme('software-update-available'),
                                          "Check for &Updates Now...", self)
            action_check_update.triggered.connect(lambda: pysparkle.check_update(verbose=True, force=True))
            CONFIG.set_shortcut(action_check_update, 'check_for_updates', 'Check for Updates')
            self.menu.insertAction(self._pysparkle_place, action_check_update)

        self.shown.disconnect(self.init_pysparkle)

    class AboutWindow(QDialog):

        def __init__(self, text, parent=None):
            super().__init__(parent)
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Popup)
            self.setWindowTitle("About PLaSK")
            self.setStyleSheet("""
                QDialog {
                    border-style: outset;
                    border-width: 1px;
                    border-color: black;
                }
            """)
            outer = QVBoxLayout()
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            try:
                scale = QApplication.primaryScreen().physicalDotsPerInch() / 96.
            except AttributeError:
                scale = QApplication.desktop().physicalDpiX() / 96.
            if scale < 1.4:
                image_name = 'splash620'
            elif scale < 1.8:
                image_name = 'splash868'
            else:
                image_name = 'splash1116'

            image_label = QLabel()
            image_label.setPixmap(QPixmap(os.path.join(os.path.dirname(__file__), image_name)))
            outer.addWidget(image_label)

            vertical = QVBoxLayout()
            vertical.setContentsMargins(16, 8, 16, 8)

            horizontal = QHBoxLayout()
            horizontal.setContentsMargins(0, 0, 0, 0)
            horizontal.setSpacing(16)

            style = QApplication.style()
            ics = style.pixelMetric(QStyle.PixelMetric.PM_MessageBoxIconSize)
            icon = QIcon.fromTheme('dialog-information')
            icon_label = QLabel()
            icon_label.setPixmap(icon.pixmap(ics, QIcon.Mode.Normal, QIcon.State.Off))
            icon_label.setFixedWidth(ics)
            horizontal.addWidget(icon_label)
            horizontal.setAlignment(icon_label, Qt.AlignmentFlag.AlignTop)

            if VERSION is not None:
                year = VERSION[:4]
            else:
                year = date.today().strftime("%Y")

            label = QLabel(f"<b>PLaSK — Photonic Laser Simulation Kit</b><br/>\n"
                           f"© 2014-{year} Lodz University of Technology, Photonics Group<br/><br/>" + text)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            horizontal.addWidget(label)

            vertical.addLayout(horizontal)

            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
            button_box.accepted.connect(self.accept)
            vertical.addWidget(button_box)

            inner = QWidget()
            inner.setLayout(vertical)
            outer.addWidget(inner)

            self.setLayout(outer)

            self.adjustSize()

    def about(self):

        if VERSION is not None:
            details = "Version <b>{}</b><br/>(GUI using {} framework)<br/>\n<br/>\n".format(VERSION, QT_API)
        else:
            details = ""
        details += "This program is distributed under GPL3 License and in hope that<br>\n"\
                   "it will be useful, but WITHOUT ANY WARRANTY; without even the<br/>\n"\
                   "implied warranty of MERCHANTABILITY or FITNESS FOR ANY PURPOSE."
        note = '<br/>\n<br/>\n<span style="color: #888888;">Details have been copied to ' \
               'your clipboard.</span>'

        if pysparkle is not None:
            from pysparkle.backend import appcast
            details += "<br/>\n<br/>\nOperating System: {}<br/>\n{}" \
                .format(appcast.OS.title(), ", ".join(appcast.DISTS))

        msgbox = self.AboutWindow(details + note, self)

        details = re.sub('<[^>]+>', '', details).replace('&lt;', '<').replace('&gt;', '>')
        QApplication.clipboard().setText(details)

        msgbox.move(self.frameGeometry().topLeft() + self.rect().center() - msgbox.rect().center())
        qt_exec(msgbox)

    def toggle_fullscreen(self):
        state = self.windowState()
        if not state & Qt.WindowState.WindowFullScreen:
            self.setWindowState(state | Qt.WindowState.WindowFullScreen)
        else:
            self.setWindowState(state & ~Qt.WindowState.WindowFullScreen)

class GotoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Go to Line")
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        label = QLabel("Line number:")
        self.input = QLineEdit()
        self.input.setValidator(QIntValidator(self.input))
        hbox.addWidget(label)
        hbox.addWidget(self.input)
        vbox.addLayout(hbox)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)
        self.setLayout(vbox)


try:
    VERSION = plask.version
except (NameError, AttributeError):
    from .utils.files import which
    try:
        plask_exe = which('plask') or 'plask'
        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags = subprocess.STARTF_USESTDHANDLES | subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
        except AttributeError:
            proc = subprocess.Popen([plask_exe, '-V'], stdout=subprocess.PIPE)
        else:
            proc = subprocess.Popen([plask_exe, '-V'], startupinfo=si, stdout=subprocess.PIPE)
        info, err = proc.communicate()
        VERSION = info.decode('utf8').strip()
    except:
        VERSION = None


def _handle_exception(exc_type, exc_value, exc_traceback):
    global error_file
    if exc_type == SystemExit:
        sys.exit(exc_value.code)
    else:
        dat = u''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + '\n'
        if error_file is None:
            if _DEBUG:
                msg = dat
            else:
                msg = exc_type.__name__ + ": " + str(exc_value)
        else:
            msg = None

        if _DEBUG and os.name != 'nt':
            out = sys.stderr
        else:
            if error_file is None:
                import time
                error_file = os.path.join(CURRENT_DIR,
                                          time.strftime("plaskgui.%Y%m%d.%H%M%S.error.log",
                                                        time.localtime(time.time())))
                msg += u"\n\nError details were saved to:\n{}".format(error_file)
                msg += u"\n\nFurther errors will not be reported (but they will be saved to the above file)."
            out = open(error_file, 'a')
        out.write(dat)
        out.flush()

        if msg is not None:
            if os.name == 'nt':
                import ctypes
                MessageBox = ctypes.windll.user32.MessageBoxA
                MessageBox(None, msg.encode('utf-8'), u"PLaSK GUI Error".encode('utf-8'), 0x10)
            elif not _DEBUG:
                try:
                    QMessageBox.critical(None, "PLaSK GUI Error", msg)
                except:
                    pass


error_file = None
sys.excepthook = _handle_exception


PLUGINS = []

def load_plugins():
    global PLUGINS

    name_re = re.compile(r'^#\s*plugin:\s*(.*)\s*$', re.IGNORECASE)
    desc_re = re.compile(r'^#\s*desc(?:ription)?:\s*(.*)\s*$', re.IGNORECASE)

    if os.name == 'nt':
        plugin_dirs = [os.path.join(os.environ.get('SYSTEMDRIVE', 'C:'), r'ProgramData\PLaSK\gui\plugins')]
        if 'LOCALAPPDATA' in os.environ:
            plugin_dirs.insert(0, os.path.join(os.environ['LOCALAPPDATA'], r"PLaSK\gui\plugins"))
    else:
        plugin_dirs = [os.path.expanduser("~/.local/lib/plask/gui/plugins"), "/etc/plask/gui/plugins"]
    plugin_dirs.append(os.path.join(__path__[0], 'plugins'))

    for finder, modname, ispkg in pkgutil.iter_modules(plugin_dirs):
        name = desc = None
        if sys.version_info < (3, 5):
            fname = finder.find_module(modname).get_filename()
        else:
            fname = finder.find_spec(modname).loader.path
        if ispkg and not fname.endswith('__init__.py'):
            fname = os.path.join(fname, '__init__.py')
        try:
            for line in open(fname):
                line = line.strip()
                m = name_re.match(line)
                if m is not None:
                    name = m.group(1)
                m = desc_re.match(line)
                if m is not None:
                    desc = m.group(1) if desc is None else desc + '<br/>' + m.group(1)
                if name is not None and desc is not None and not line.startswith('#'): break
        except:
            pass
        if name is not None:
            PLUGINS.append((modname, name, desc))
            try:
                if CONFIG.get('plugins/{}'.format(modname), True):
                    if sys.version_info < (3, 5):
                        finder.find_module(modname).load_module(modname)
                    else:
                        fname = finder.find_spec(modname).loader.load_module(modname)
            except:
                if _DEBUG:
                    import traceback as tb
                    tb.print_exc()


class Session:
    def __init__(self):
        self.opened_files = []

    def commit(self, session_manager):
        self.opened_files = [window.document.filename for window in WINDOWS]

    def save(self, session_manager):
        # self.opened_files.extend(window.document.filename for window in WINDOWS)  # if some windows are still open
        args = [APPLICATION.arguments()[0]] + self.opened_files
        session_manager.setRestartCommand(args)


def main():
    try:
        _debug_index = sys.argv.index('-debug')
    except ValueError:
        pass
    else:
        global _DEBUG
        del sys.argv[_debug_index]
        _DEBUG = True

    if _DEBUG:
        sys.stderr.write("PLaSK GUI, version {}.\nUsing {} API.\n".format(VERSION, QT_API))
        import faulthandler
        faulthandler.enable()

    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        pass

    global APPLICATION, SESSION, pysparkle

    APPLICATION = QApplication(sys.argv)
    APPLICATION.setApplicationName("PLaSK")
    APPLICATION.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)
    sys.argv = APPLICATION.arguments()

    prepare_completions()

    if CONFIG['main_window/dark_style']:
        APPLICATION.setStyle(QStyleFactory.create("Fusion"))
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor('#dddddd'))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor('#dddddd'))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor('#dddddd'))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor('#dddddd'))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor('#dddddd'))
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        APPLICATION.setPalette(dark_palette)
        APPLICATION.setStyleSheet("QToolTip { color: #dddddd; background-color: #2a82da; border: 1px solid #dddddd; }")

    pysparkle = None

    icons_theme = str(CONFIG['main_window/icons_theme']).lower()
    icons_path = QIcon.themeSearchPaths()[:-1]
    if icons_theme == 'system':
        icons_theme = QIcon.themeName()
        if not icons_theme:
            if dark_style():
                QIcon.setThemeName('breeze-dark-plask')
            else:
                QIcon.setThemeName('breeze-plask')
        icons_path.append(os.path.join(__path__[0], 'icons'))
    else:
        if icons_theme == 'tango': icons_theme = 'hicolor'
        elif icons_theme == 'breeze':
            if dark_style():
                icons_theme = 'breeze-dark-plask'
            else:
                icons_theme = 'breeze-plask'
        icons_path.insert(0, os.path.join(__path__[0], 'icons'))
    QIcon.setThemeSearchPaths(icons_path)
    QIcon.setThemeName(icons_theme)

    load_plugins()

    if matplotlib:
        font = APPLICATION.font()
        try:
            dpi = APPLICATION.primaryScreen().logicalDotsPerInch()
        except AttributeError:
            dpi = APPLICATION.desktop().logicalDpiY()
        color = APPLICATION.palette().color(QPalette.ColorRole.Text).name()
        matplotlib.rcParams.update({
            'figure.dpi': dpi,
            'font.family': font.family(),
            'font.size': font.pointSize(),
            'font.weight': 'normal',
            'mathtext.fontset': 'custom',
            'mathtext.cal': font.family(),
            'mathtext.rm': font.family(),
            'mathtext.it': font.family() + ':italic',
            'mathtext.bf': font.family() + ':bold',
            'mathtext.sf': font.family(),
            'text.color': color,
            'axes.labelcolor': color,
            'xtick.color': color,
            'ytick.color': color,
            # 'patch.edgecolor': color,
            # 'axes.edgecolor': color,
            # 'grid.color': color,
            'figure.facecolor': APPLICATION.palette().color(QPalette.ColorRole.Window).name(),
        })
        if CONFIG['workarounds/no_unicode_minus']:
            try:
                matplotlib.rcParams['axes.unicode_minus'] = False
            except KeyError:
                pass

    SESSION = Session()

    files = [os.path.abspath(arg) for arg in sys.argv[1:]]
    if files:
        for filename in files:
            WINDOWS.add(MainWindow(filename))
    else:
        WINDOWS.add(MainWindow())

    APPLICATION.commitDataRequest.connect(SESSION.commit)
    APPLICATION.saveStateRequest.connect(SESSION.save)

    try:
        plask._plask._close_splash()
    except (NameError, AttributeError):
        pass

    exit_code = qt_exec(APPLICATION)

    try:
        # This prevents crash on exit, because of PyFinalize not being supported by boost
        plask.material.db.clear()
    except NameError:
        pass

    sys.exit(exit_code)
