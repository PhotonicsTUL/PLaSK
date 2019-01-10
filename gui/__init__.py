#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import pkgutil
import traceback
import datetime

from lxml import etree
from uuid import getnode

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import gui
    gui.main()

ACTIONS = []
HELP_ACTIONS = []

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str

try:
    _DEBUG = bool(eval(os.environ['PLASKGUI_DEBUG'].title()))
except KeyError:
    _DEBUG = False

from .qt.QtCore import *
from .qt.QtWidgets import *
from .qt.QtGui import *
from .qt import QtSignal, QT_API

sys.path.insert(2, os.path.join(__path__[0], 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'plask', 'stubs'))

sys.path.insert(4, os.path.join(__path__[0], 'external', 'pysparkle'))


# Set-up correct backend for matplotlib
try:
    import matplotlib
except ImportError:
    matplotlib = None

from .xpldocument import XPLDocument
from .pydocument import PyDocument
from .model.info import InfoListModel, Info
from .launch import launch_plask
from .controller.materials.plot import show_material_plot

from .utils.config import CONFIG, ConfigProxy, ConfigDialog
from .utils.texteditor import update_textedit_colors
from .utils.widgets import fire_edit_end, InfoListView
from .utils.help import open_help

try:
    from pysparkle import PySparkle
except:
    PySparkle = None

try:
    import plask
except ImportError:
    pass


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


# icons: http://standards.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
SECTION_ICONS = {
    'defines': 'accessories-dictionary',
    'materials': 'accessories-character-map',
    'geometry': 'system-file-manager',
    'grids': 'preferences-desktop-keyboard',
    'solvers': 'utilities-system-monitor',
    'connects': 'preferences-desktop-accessibility',
    'script': 'accessories-text-editor',
}


class MainWindow(QMainWindow):

    SECTION_TITLES = dict(defines=" &Defines ", materials=" &Materials ", geometry=" &Geometry ", grids=" M&eshing ",
                          solvers=" &Solvers ", connects=" &Connects ", script=" Sc&ript ")

    SECTION_TIPS = {
        'defines': "Edit the list of pre-defined variables for use in the rest of the file (Alt+D)",
        'materials': "Edit custom materials (Alt+M)",
        'geometry': "Edit geometries of your structures (Alt+G)",
        'grids': "Edit computational meshes or set-up automatic mesh generators (Alt+E)",
        'solvers': "Create and configure computational solvers (Alt+S)",
        'connects': "Define connections between computational solvers (Alt+C)",
        'script': "Edit control script for your computations (Alt+R)"}

    shown = QtSignal()
    closing = QtSignal(QCloseEvent)
    closed = QtSignal()
    config_changed = QtSignal()

    def __init__(self, filename=None, Document=XPLDocument):
        super(MainWindow, self).__init__()

        self.document = None

        self.current_tab_index = -1
        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged[int].connect(self.tab_change)
        # self.tabs.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

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
            'Show Sour&ce', self)
        self.showsource_action.setShortcut(QKeySequence(Qt.Key_F4))
        self.showsource_action.setCheckable(True)
        self.showsource_action.setStatusTip('Show XPL source of the current section')
        self.showsource_action.setEnabled(False)

        self.setWindowIcon(QIcon.fromTheme('plaskgui'))

        self.info_model = InfoListModel(None)
        self.info_table = InfoListView(self.info_model, self)
        self.info_table.setModel(self.info_model)
        self.info_table.setSelectionMode(QListView.NoSelection)
        self.info_model.entries = [Info('', Info.NONE)]
        self.info_table.setFixedHeight(self.info_table.sizeHintForRow(0))
        self.info_model.entries = []
        info_selection_model = self.info_table.selectionModel()
        info_selection_model.currentChanged.connect(self._on_select_info)

        self.info_table.setFrameShape(QFrame.NoFrame)
        layout.addWidget(self.info_table)
        self.info_model.layoutChanged.connect(self._update_info_color)

        if filename is None or not self._try_load_from_file(filename):  # try to load only if filename is not None
            self.document = Document(self)
            self.setup_model()
            self.setWindowTitle("[*] PLaSK")
            self.setWindowModified(False)

        new_action = QAction(QIcon.fromTheme('document-new'), '&New', self)
        new_action.setShortcut(QKeySequence.New)
        new_action.setStatusTip('Create a new XPL file')
        new_action.triggered.connect(lambda: self.new(XPLDocument))

        newpy_action = QAction(QIcon.fromTheme('document-new'), 'New &Python', self)
        newpy_action.setStatusTip('Create a new Python file')
        newpy_action.triggered.connect(lambda: self.new(PyDocument))

        open_action = QAction(QIcon.fromTheme('document-open'), '&Open...', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip('Open an existing file')
        open_action.triggered.connect(self.open)

        save_action = QAction(QIcon.fromTheme('document-save'), '&Save', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.setStatusTip('Save the file to disk')
        save_action.triggered.connect(self.save)

        saveas_action = QAction(QIcon.fromTheme('document-save-as'), 'Save &As...', self)
        saveas_action.setShortcut(QKeySequence.SaveAs)
        saveas_action.setStatusTip('Save the file to disk asking for a new name')
        saveas_action.triggered.connect(self.save_as)

        reload_action = QAction(QIcon.fromTheme('view-refresh'), '&Reload', self)
        reload_action.setStatusTip('Reload the current file from disk')
        reload_action.triggered.connect(self.reload)

        launch_action = QAction(QIcon.fromTheme('media-playback-start'), '&Launch...', self)
        launch_action.setShortcut('F5')
        launch_action.setStatusTip('Launch the current file in PLaSK')
        launch_action.triggered.connect(lambda: launch_plask(self))

        goto_action = QAction(QIcon.fromTheme('go-jump'), '&Go to Line...', self)
        goto_action.setShortcut(Qt.CTRL + Qt.Key_L)
        goto_action.setStatusTip('Go to the specified line')
        goto_action.triggered.connect(self.on_goto_line)

        plot_material_action = QAction(QIcon.fromTheme('matplotlib'), 'Examine &Material Parameters...', self)
        plot_material_action.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_M)
        plot_material_action.triggered.connect(lambda: show_material_plot(self, self.document.materials.model,
                                                                          self.document.defines.model))

        settings_action = QAction(QIcon.fromTheme('document-properties'), 'GUI Se&ttings...', self)
        settings_action.setStatusTip('Change some GUI settings')
        settings_action.triggered.connect(self.show_settings)

        about_action = QAction(QIcon.fromTheme('dialog-information'), 'A&bout...', self)
        about_action.setStatusTip('Show information about PLaSK')
        about_action.triggered.connect(self.about)

        help_action = QAction(QIcon.fromTheme('help-contents'), 'Open &Help...', self)
        help_action.setStatusTip('Open on-line help in a web browser')
        help_action.triggered.connect(lambda: open_help(main_window=self))

        install_license_action = QAction('Install License...', self)
        install_license_action.setStatusTip('Install PLaSK license file into a proper location')
        install_license_action.triggered.connect(self.install_license)

        exit_action = QAction(QIcon.fromTheme('application-exit'), 'E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
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
        self.menu.addAction(install_license_action)
        self._pysparkle_place = self.menu.addSeparator()
        self.menu.addAction(settings_action)
        self.menu.addSeparator()
        self.menu.addAction(exit_action)

        if os.name == 'nt':
            menu_button = QToolButton(self)
            menu_button.setPopupMode(QToolButton.InstantPopup)
            menu_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            menu_button.setAutoFillBackground(True)
            font = menu_button.font()
            font.setBold(True)
            menu_button.setFont(font)
        else:
            menu_button = QPushButton(self)
        menu_button.setText("PLaSK")
        pal = menu_button.palette()
        pal.setColor(QPalette.Button, QColor("#88aaff"))
        menu_button.setIcon(QIcon.fromTheme('plask-logo'))
        menu_button.setPalette(pal)
        menu_button.setToolTip("Show operations menu (F2)")
        menu_shortcut = QShortcut(QKeySequence(Qt.Key_F2), self)
        menu_shortcut.activated.connect(menu_button.showMenu)

        menu_button.setMenu(self.menu)
        self.tabs.setCornerWidget(menu_button, Qt.TopLeftCorner)

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
        self.tabs.setCornerWidget(source_button, Qt.TopRightCorner)

        self.shown.connect(self.init_pysparkle, Qt.QueuedConnection)

        fs = int(1.3 * QFont().pointSize())
        self.tabs.setStyleSheet("QTabBar {{ font-size: {}pt; }}".format(fs))
        menu_button.setStyleSheet("QPushButton {{ font-size: {}pt; font-weight: bold; }}".format(fs))

        self.config_changed.connect(update_textedit_colors)

        desktop = QDesktopWidget()
        geometry = CONFIG['session/geometry']
        if geometry is not None:
            screen = desktop.availableGeometry(geometry.center())
            if geometry.left() <= screen.left()+1 and geometry.top() <= screen.top()+1 and \
               geometry.right()+1 >= screen.width() and geometry.bottom()+1 >= screen.height():
                self.showMaximized()
            else:
                geometry.setWidth(min(geometry.width(), screen.right()-geometry.left()+1))
                geometry.setHeight(min(geometry.height(), screen.bottom()-geometry.top()+1))
                self.setGeometry(geometry)
        else:
            screen = desktop.availableGeometry(self)
            self.resize(screen.width() * 0.8, screen.height() * 0.9)

        self.setAcceptDrops(True)

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
        super(MainWindow, self).showEvent(event)
        self.shown.emit()

    def _update_info_color(self):
        pal = self.info_table.palette()
        if any(info.level != Info.NONE for info in self.info_model.entries):
            pal.setColor(QPalette.Base, QColor("#ffc"))
        else:
            pal.setColor(QPalette.Base, pal.color(QPalette.Window))
        self.info_table.setPalette(pal)

    def _on_select_info(self, current, _):
        if not current.isValid(): return
        self.current_controller.select_info(self.info_model.entries[current.row()])
        self.info_table.setCurrentIndex(QModelIndex())

    @property
    def current_controller(self):
        if self.current_tab_index == -1: return None
        return self.document.controller_by_index(self.current_tab_index)

    def update_recent_menu(self):
        self.recent_menu.clear()
        class Func(object):
            def __init__(s, f): s.f = f
            def __call__(s): return self.open(s.f)
        for i,f in enumerate(reversed(load_recent_files())):
            action = QAction(f, self)
            action.triggered.connect(Func(f))
            # action.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_0 + (i+1)%10))
            self.recent_menu.addAction(action)

    def _try_load_from_file(self, filename, tab=None):
        update_recent_files(os.path.abspath(filename))
        document = PyDocument(self) if filename.endswith('.py') else XPLDocument(self)
        try:
            document.load_from_file(filename)
        except Exception as e:
            if _DEBUG: raise e
            QMessageBox.critical(self, 'File load error',
                                       'Error while loading file "{}":\n{}'.format(filename, str(e)))
            return False
        else:
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
        if isinstance(self.document, PyDocument):
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

    def reload(self):
        if self.document.filename is None:
            return
        if self.isWindowModified():
            confirm = QMessageBox.question(self, "Unsaved Changes",
                                           "File has unsaved changes. Do you want to discard them and reload the file?",
                                           QMessageBox.Yes | QMessageBox.No,  QMessageBox.No)
            if confirm == QMessageBox.No:
                return
        current_tab_index = self.current_tab_index
        self.document.controller_by_index(current_tab_index).on_edit_exit()
        self.current_tab_index = -1
        if not self._try_load_from_file(self.document.filename, current_tab_index):
            self.setWindowModified(True)

    def _save_document(self, filename):
        fire_edit_end()
        try:
            self.document.save_to_file(unicode(filename))
        except Exception as err:
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Save Error")
            msgbox.setText("The file '{}' could not be saved to disk.".format(filename))
            msgbox.setInformativeText(unicode(err))
            msgbox.setStandardButtons(QMessageBox.Ok)
            msgbox.setIcon(QMessageBox.Critical)
            msgbox.exec_()
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
        """Ask for filename and save to chosen file. Return true only when file has been saved."""
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
        if self.current_tab_index != -1:
            try:
                self.document.controller_by_index(self.current_tab_index).save_data_in_model()
            except Exception as e:
                msgbox = QMessageBox()
                msgbox.setText("Edited content of the current section is invalid.")
                msgbox.setDetailedText(str(e))
                msgbox.setInformativeText("Do you want to save anyway (with the old content of the current section)?")
                msgbox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msgbox.setIcon(QMessageBox.Warning)
                #msgbox.setDefaultButton(QMessageBox.Yes);
                return msgbox.exec_() == QMessageBox.Yes
        errors = self.document.get_info(Info.ERROR)
        if errors:
            msgbox = QMessageBox()
            msgbox.setText("Document contains some non-critical errors.\n\n"
                           "It is possible to save it, however launching it will most probably fail.")
            msgbox.setDetailedText(u'\n'.join(map(unicode, errors)))
            msgbox.setInformativeText("Do you want to save anyway?")
            msgbox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msgbox.setIcon(QMessageBox.Warning)
            msgbox.setDefaultButton(QMessageBox.Yes)
            return msgbox.exec_() == QMessageBox.Yes
        return True

    def show_settings(self):
        dialog = ConfigDialog(self)
        dialog.exec_()

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
            self.info_model.setModel(c.model)
            c.on_edit_enter()
        else:
            self.info_model.setModel(None)

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

        self.closing.emit(event)
        if not event.isAccepted():
            return

        if self.isWindowModified():
            confirm = QMessageBox.question(self, "Unsaved File",
                                                 "File is not saved. Do you want to save it before closing the window?",
                                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if confirm == QMessageBox.Cancel or (confirm == QMessageBox.Yes and not self.save()):
                event.ignore()
                return

        self.closed.emit()
        self.document = None

        try:
            WINDOWS.remove(self)
        except KeyError:
            pass

        geometry = self.geometry()
        CONFIG['session/geometry'] = geometry
        CONFIG.sync()

    def set_changed(self, changed):
        """Set changed flags in the document window"""
        try:
            if self.document.filename:
                self.setWindowTitle(u"{}[*] - PLaSK".format(self.document.filename))
            else:
                self.setWindowTitle("[*] PLaSK")
        except AttributeError:
            self.setWindowTitle("[*] PLaSK")
        self.setWindowModified(changed)

    def on_goto_line(self):
        dialog = GotoDialog(self)
        if dialog.exec_():
            self.goto_line(int(dialog.input.text()))
        else:
            return

    def goto_line(self, line_number=None):
        if line_number is None:
            dialog = GotoDialog(self)
            if dialog.exec_():
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
                                      frontend='qt5' if QT_API == 'PyQt5' else 'qt4')
        if pysparkle is not None:
            action_check_update = QAction(QIcon.fromTheme('software-update-available'),
                                          "Check for &Updates Now...", self)
            action_check_update.triggered.connect(lambda: pysparkle.check_update(verbose=True, force=True))
            self.menu.insertAction(self._pysparkle_place, action_check_update)
        self.shown.disconnect(self.init_pysparkle)

    class AboutWindow(QDialog):

        def __init__(self, text, parent=None):
            super(MainWindow.AboutWindow, self).__init__(parent)
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup)
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

            scale = QApplication.desktop().logicalDpiX() / 96.
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
            ics = style.pixelMetric(QStyle.PM_MessageBoxIconSize)
            icon = QIcon.fromTheme('dialog-information')
            icon_label = QLabel()
            icon_label.setPixmap(icon.pixmap(ics, QIcon.Normal, QIcon.Off))
            icon_label.setFixedWidth(ics)
            horizontal.addWidget(icon_label)
            horizontal.setAlignment(icon_label, Qt.AlignTop)

            label = QLabel(u"<b>PLaSK — Photonic Laser Simulation Kit</b><br/>\n"
                           u"© 2014-2019 Lodz University of Technology, Photonics Group<br/><br/>" + text)
            label.setTextFormat(Qt.RichText)
            label.setWordWrap(True)
            horizontal.addWidget(label)

            vertical.addLayout(horizontal)

            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(self.accept)
            vertical.addWidget(button_box)

            inner = QWidget()
            inner.setLayout(vertical)
            outer.addWidget(inner)

            self.setLayout(outer)

            self.adjustSize()

    def about(self):

        if VERSION is not None:
            details = u"Version <b>" + VERSION + u"</b><br/>(GUI using {} framework)<br/>\n<br/>\n".format(QT_API)
        else:
            details = ""
        user = LICENSE.get('user', '')
        institution = LICENSE.get('institution', '')
        if user or institution:
            try: user = user.decode('utf8')
            except AttributeError: pass
            if institution:
                try: institution = institution.decode('utf8')
                except AttributeError: pass
                institution = "<br/>\n" + institution.replace('<', '&lt;').replace('>', '&gt;')
            details += u"Licensed to:<br/>\n{}{}".format(
                user.replace('<', '&lt;').replace('>', '&gt;'), institution)
            if 'expiration' in LICENSE:
                date = LICENSE['expiration'].strftime('%x')
                try: date = date.decode('utf8')
                except AttributeError: pass
                details += u"<br/>\nLicense active until " + date
            details += '<br/>\n'
        details += "Your system ID is {}".format(LICENSE['systemid'])
        note = '<br/>\n<br/>\n<span style="color: #888888;">Details have been copied to ' \
               'your clipboard.</span>'

        msgbox = self.AboutWindow(details + note, self)

        details = re.sub('<[^>]+>', '', details).replace('&lt;', '<').replace('&gt;', '>')
        QApplication.clipboard().setText(details)

        msgbox.move(self.frameGeometry().topLeft() + self.rect().center() - msgbox.rect().center())
        msgbox.exec_()

    def install_license(self):
        filename = QFileDialog.getOpenFileName(self, "Open file", CURRENT_DIR,
                                               "PLaSK license file (plask_license.xml)")
        if type(filename) == tuple: filename = filename[0]
        if not filename: return
        from shutil import copy
        dest = os.path.join(os.environ['USERPROFILE'], "plask_license.xml") if os.name == 'nt' else \
               os.path.expanduser("~/.plask_license.xml")
        if os.path.lexists(dest):
            msgbox = QMessageBox()
            msgbox.setWindowTitle("License Exists")
            msgbox.setText("The license file '{}' already exists. Do you want to replace it?".format(dest))
            msgbox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msgbox.setIcon(QMessageBox.Question)
            answer = msgbox.exec_()
            if answer == QMessageBox.No: return
        try:
            copy(filename, dest)
        except Exception as err:
            msgbox = QMessageBox()
            msgbox.setWindowTitle("License Install Error")
            msgbox.setText("The license file '{}' could not be installed.".format(filename))
            msgbox.setInformativeText(unicode(err))
            msgbox.setStandardButtons(QMessageBox.Ok)
            msgbox.setIcon(QMessageBox.Critical)
            msgbox.exec_()
        else:
            dom = etree.parse(filename)
            root = dom.getroot()
            xns = root.nsmap.get(None, '')
            if xns: xns = '{' + xns + '}'
            if root.tag != xns+'license': return
            data = root.find(xns+'data')
            name = data.find('name')
            if name is not None: name = name.text
            email = data.find('email')
            if email is not None:
                if name is None: LICENSE['user'] = email.text
                else: LICENSE['user'] = (name + " <" + email.text + ">")
            institution = data.find('institution')
            if institution is not None:
                LICENSE['institution'] = institution.text
            elif 'institution' in LICENSE:
                del LICENSE['institution']
            expiry = data.find('expiry')
            if expiry is not None:
                LICENSE['expiration'] = _parse_expiry(expiry.text)
            elif 'expiration' in LICENSE:
                del LICENSE['expiration']
            self.about()


class GotoDialog(QDialog):
    def __init__(self, parent=None):
        super(GotoDialog, self).__init__(parent)
        self.setWindowTitle("Go to Line")
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        label = QLabel("Line number:")
        self.input = QLineEdit()
        self.input.setValidator(QIntValidator(self.input))
        hbox.addWidget(label)
        hbox.addWidget(self.input)
        vbox.addLayout(hbox)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)
        self.setLayout(vbox)


def _parse_expiry(expiry):
    match = re.match('(\d+).(\d+).(\d+)', expiry)
    if match is not None:
        return datetime.date(*reversed([int(g) for g in match.groups()]))


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
        info = info.decode('utf8').strip()
        if '\n' in info:
            VERSION, LICENSE = info.split("\n", 1)
            _, VERSION = VERSION.split(' ', 1)
            user, expiry = LICENSE.rsplit(' ', 1)
            LICENSE = dict(user=user)
            expiry = _parse_expiry(expiry)
            if expiry is not None:
                LICENSE['expiration'] = expiry
        else:
            VERSION, LICENSE = info, {}
    except:
        VERSION, LICENSE = None, {}
else:
    try:
        LICENSE = plask.license
    except (AttributeError, NameError):
        LICENSE = dict(user='', date='')
if 'systemid' not in LICENSE or not LICENSE['systemid']:
    LICENSE['systemid'] = "{:X}".format(getnode())[-12:]


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
                msg += u"\n\nError details saved to:\n{}".format(error_file)
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

    name_re = re.compile(r'^\s*#\s*plugin:\s*(.*)\s*$', re.IGNORECASE)
    desc_re = re.compile(r'^\s*#\s*description:\s*(.*)\s*$', re.IGNORECASE)

    if os.name == 'nt':
        plugin_dirs = [os.path.join(os.environ.get('SYSTEMDRIVE', 'C:'), r'ProgramData\PLaSK\gui\plugins')]
        if 'LOCALAPPDATA' in os.environ:
            plugin_dirs.insert(0, os.path.join(os.environ['LOCALAPPDATA'], r"PLaSK\gui\plugins"))
    else:
        plugin_dirs = [os.path.expanduser("~/.local/lib/plask/gui/plugins"), "/etc/plask/gui/plugins"]
    plugin_dirs.append(os.path.join(__path__[0], 'plugins'))

    for loader, modname, ispkg in pkgutil.iter_modules(plugin_dirs):
        name = desc = None
        if ispkg:
            fname = os.path.join(loader.find_module(modname).get_filename())
            if not fname.endswith('__init__.py'):
                fname = os.path.join(fname, '__init__.py')
        else:
            fname = loader.find_module(modname).get_filename()
        try:
            for line in open(fname):
                m = name_re.match(line)
                if m is not None: name = m.group(1)
                m = desc_re.match(line)
                if m is not None: desc = m.group(1)
                if name is not None and desc is not None: break
        except:
            pass
        if name is not None:
            PLUGINS.append((modname, name, desc))
            try:
                if CONFIG.get('plugins/{}'.format(modname), True):
                    loader.find_module(modname).load_module(modname)
            except:
                if _DEBUG:
                    import traceback as tb
                    tb.print_exc()



class Session(object):
    def __init__(self):
        self.opened_files = []

    def commit(self, session_manager):
        self.opened_files = [window.document.filename for window in WINDOWS]

    def save(self, session_manager):
        self.opened_files.extend(window.document.filename for window in WINDOWS)  # if some windows are still open
        if self.opened_files:
            CONFIG['session/saved_' + session_manager.sessionKey()] = self.opened_files
            CONFIG.sync()
            self.opened_files = []

    @staticmethod
    def restore():
        key = 'session/saved_' + APPLICATION.sessionKey()
        files = CONFIG[key]
        del CONFIG[key]
        ok = False
        if files:
            if type(files) is not list: files = [files]
            for file in files:
                try:
                    WINDOWS.add(MainWindow(file))
                except:
                    pass
                else:
                    ok = True
        return ok


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

    global APPLICATION, SESSION, pysparkle

    APPLICATION = QApplication(sys.argv)
    APPLICATION.setApplicationName("PLaSK")
    APPLICATION.setAttribute(Qt.AA_DontShowIconsInMenus, False)
    try:
        APPLICATION.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        APPLICATION.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        pass
    sys.argv = APPLICATION.arguments()

    pysparkle = None

    icons_theme = str(CONFIG['main_window/icons_theme']).lower()
    if icons_theme == 'system':
        icons_path = QIcon.themeSearchPaths()[:-1]
        if not QIcon.themeName():
            QIcon.setThemeName('breeze')
    else:
        if icons_theme == 'tango': icons_theme = 'hicolor'
        icons_path = []
        QIcon.setThemeName(icons_theme)
    icons_path.insert(0, os.path.join(__path__[0], 'icons'))
    QIcon.setThemeSearchPaths(icons_path)

    load_plugins()

    if matplotlib:
        ft = QWidget().font()
        pd = APPLICATION.desktop()
        matplotlib.rcParams.update({
            'figure.dpi': pd.logicalDpiY(),
            'font.size': ft.pointSize(),
            'font.family': ft.family(),
            'mathtext.fontset': 'stixsans',
            # 'mathtext.cal': ft.family(),
            # 'mathtext.rm': ft.family(),
            # 'mathtext.it': ft.family() + ':italic',
            # 'mathtext.bf': ft.family() + ':bold',
            # 'mathtext.sf': ft.family()
        })

    SESSION = Session()
    if APPLICATION.isSessionRestored():
        if not SESSION.restore():
            WINDOWS.add(MainWindow())
    else:
        if len(sys.argv) > 1:
            filename = os.path.abspath(sys.argv[1])
            WINDOWS.add(MainWindow(filename))
        else:
            WINDOWS.add(MainWindow())

    APPLICATION.commitDataRequest.connect(SESSION.commit)
    APPLICATION.saveStateRequest.connect(SESSION.save)

    try:
        plask._plask._close_splash()
    except (NameError, AttributeError):
        pass

    exit_code = APPLICATION.exec_()

    sys.exit(exit_code)
