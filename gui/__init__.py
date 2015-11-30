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
OPERATIONS = []

import sys
import os
import ctypes
import subprocess
import pkgutil

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
    _DEBUG = eval(os.environ['PLASKGUI_DEBUG'].title())
except KeyError:
    _DEBUG = False

from .qt import QtGui, QtCore, QtSignal
from .qt.QtCore import Qt

sys.path.insert(2, os.path.join(__path__[0], 'external'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'plask', 'stubs'))

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

from .utils.config import CONFIG, ConfigDialog
from .utils.texteditor import update_textedit_colors
from .utils.widgets import fire_edit_end, InfoListView

try:
    import plask
except ImportError:
    pass


try:
    winsparkle = ctypes.CDLL('WinSparkle.dll')
except OSError:
    winsparkle = None


WINDOWS = set()


CURRENT_DIR = CONFIG['session/recent_dir']
if CURRENT_DIR is None:
    CURRENT_DIR = os.getcwd()


RECENT = CONFIG['session/recent_files']
if RECENT is None:
    RECENT = []
elif type(RECENT) is not list:
    RECENT = [RECENT]


def update_recent_files(filename):
    global RECENT, CURRENT_DIR
    filename = filename.replace('/', os.path.sep)
    CURRENT_DIR = os.path.dirname(filename)
    CONFIG['session/recent_dir'] = CURRENT_DIR
    try:
        RECENT.remove(filename)
    except ValueError:
        pass
    if os.path.isfile(filename):
        RECENT.append(filename)
    RECENT = RECENT[-10:]
    CONFIG['session/recent_files'] = RECENT
    CONFIG.sync()
    for window in WINDOWS:
        window.update_recent_list()


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


class MainWindow(QtGui.QMainWindow):

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

    closing = QtSignal(QtGui.QCloseEvent)

    closed = QtSignal()

    config_changed = QtSignal()

    def __init__(self, filename=None):
        super(MainWindow, self).__init__()

        self.current_tab_index = -1
        self.tabs = QtGui.QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged[int].connect(self.tab_change)
        # self.tabs.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)

        menu_bar = QtGui.QMenuBar(self)
        menu_bar.setVisible(False)

        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs)
        main_widget = QtGui.QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.showsource_action = QtGui.QAction(
            QtGui.QIcon.fromTheme('show-source'),
            'Show Sour&ce', self)
        self.showsource_action.setShortcut(QtGui.QKeySequence(Qt.Key_F4))
        self.showsource_action.setCheckable(True)
        self.showsource_action.setStatusTip('Show XPL source of the current section')
        self.showsource_action.setEnabled(False)

        self.setWindowIcon(QtGui.QIcon.fromTheme('plaskgui'))

        self.info_model = InfoListModel(None)
        self.info_table = InfoListView(self.info_model, self)
        self.info_table.setModel(self.info_model)
        self.info_table.setSelectionMode(QtGui.QListView.NoSelection)
        self.info_model.entries = [Info('', Info.NONE)]
        self.info_table.setFixedHeight(self.info_table.sizeHintForRow(0))
        self.info_model.entries = []
        info_selection_model = self.info_table.selectionModel()
        info_selection_model.currentChanged.connect(self._on_select_info)

        self.info_table.setFrameShape(QtGui.QFrame.NoFrame)
        layout.addWidget(self.info_table)
        self.info_model.layoutChanged.connect(self._update_info_color)

        if filename is None or not self._try_load_from_file(filename):  # try to load only if filename is not None
            self.document = XPLDocument(self)
            self.setup_model()

        new_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-new'),
                                   '&New', self)
        new_action.setShortcut(QtGui.QKeySequence.New)
        new_action.setStatusTip('New XPL file')
        new_action.triggered.connect(self.new)

        open_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-open'),
                                    '&Open...', self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.setStatusTip('Open XPL file')
        open_action.triggered.connect(self.open)

        save_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-save'),
                                    '&Save', self)
        save_action.setShortcut(QtGui.QKeySequence.Save)
        save_action.setStatusTip('Save XPL file')
        save_action.triggered.connect(self.save)

        saveas_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-save-as'),
                                      'Save &As...', self)
        saveas_action.setShortcut(QtGui.QKeySequence.SaveAs)
        saveas_action.setStatusTip('Save XPL file, ask for name of file')
        saveas_action.triggered.connect(self.save_as)

        launch_action = QtGui.QAction(QtGui.QIcon.fromTheme('media-playback-start',
                                                            QtGui.QIcon(':/media-playback-start.png')),
                                      '&Launch...', self)
        launch_action.setShortcut('F5')
        launch_action.setStatusTip('Launch current file in PLaSK')
        launch_action.triggered.connect(lambda: launch_plask(self))

        goto_action = QtGui.QAction('&Go to Line...', self)
        goto_action.setShortcut(Qt.CTRL + Qt.Key_L)
        goto_action.setStatusTip('Go to specified line')
        goto_action.triggered.connect(self.goto_line)

        plot_material_action = QtGui.QAction(QtGui.QIcon.fromTheme('matplotlib'),
                               'Examine &Material Parameters...', self)
        plot_material_action.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_M)
        plot_material_action.triggered.connect(lambda: show_material_plot(self, self.document.materials.model))

        settings_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-properties'),
                                    'GUI Se&ttings...', self)
        settings_action.setStatusTip('Change some GUI settings')
        settings_action.triggered.connect(self.show_settings)

        exit_action = QtGui.QAction(QtGui.QIcon.fromTheme('application-exit'),
                                    'E&xit', self)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)

        self.recent_menu = QtGui.QMenu('Open &Recent')
        self.recent_menu.setIcon(
            QtGui.QIcon.fromTheme('document-open-recent'))
        self.update_recent_list()

        self.menu = QtGui.QMenu('&File')

        self.menu.addAction(new_action)
        self.menu.addAction(open_action)
        self.menu.addMenu(self.recent_menu)
        self.menu.addAction(save_action)
        self.menu.addAction(saveas_action)
        self.menu.addSeparator()
        self.menu.addAction(goto_action)
        self.menu.addAction(self.showsource_action)
        self.menu.addAction(launch_action)
        self.menu.addSeparator()
        self.menu.addAction(plot_material_action)
        if OPERATIONS:
            self.menu.addSeparator()
            for op in OPERATIONS:   # for plugins use
                self.menu.addAction(op(self))
        self.menu.addSeparator()
        self.menu.addAction(settings_action)
        self.menu.addSeparator()
        self.menu.addAction(exit_action)

        menu_button = QtGui.QPushButton(self)
        menu_button.setText("File")
        pal = menu_button.palette()
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#88aaff"))
        if os.name == 'nt':
            menu_button.setIcon(QtGui.QIcon(QtGui.QIcon.fromTheme('plask').pixmap(16,16)))
        else:
            menu_button.setIcon(QtGui.QIcon.fromTheme('plask'))
        menu_button.setPalette(pal)
        menu_button.setShortcut(QtGui.QKeySequence(Qt.Key_F2))
        menu_button.setToolTip("Show operations menu (F2)")

        menu_button.setMenu(self.menu)
        self.tabs.setCornerWidget(menu_button, QtCore.Qt.TopLeftCorner)

        tabs_menu = QtGui.QMenu("Sections", menu_bar)
        def add_tab_menu(indx):
            def show_tab():
                self.tabs.setCurrentIndex(indx)
                self.tab_change(indx)
            tabs_action.triggered.connect(show_tab)
        for i in range(self.tabs.count()):
            tabs_action = QtGui.QAction(self.tabs.tabText(i), self)
            add_tab_menu(i)
            tabs_menu.addAction(tabs_action)
        menu_bar.addMenu(self.menu)
        menu_bar.addMenu(tabs_menu)

        source_button = QtGui.QToolButton(self)
        source_button.setDefaultAction(self.showsource_action)
        self.tabs.setCornerWidget(source_button, QtCore.Qt.TopRightCorner)

        global winsparkle
        if winsparkle:
            if winsparkle:
                self.menu.addSeparator()
                try:
                    actionWinSparkleAutoupdate = QtGui.QAction(self)
                    actionWinSparkleAutoupdate.setText(self.tr("Automatic Updates"))
                    actionWinSparkleAutoupdate.setCheckable(True)
                    actionWinSparkleAutoupdate.setChecked(winsparkle.win_sparkle_get_automatic_check_for_updates())
                    actionWinSparkleAutoupdate.triggered.connect(
                        lambda: winsparkle.win_sparkle_set_automatic_check_for_updates(
                            int(actionWinSparkleAutoupdate.isChecked())))
                    self.menu.addAction(actionWinSparkleAutoupdate)
                except AttributeError:
                    pass
                actionWinSparkle = QtGui.QAction(self)
                actionWinSparkle.setText("Check for Updates Now...")
                actionWinSparkle.triggered.connect(lambda: winsparkle.win_sparkle_check_update_with_ui())
                self.menu.addAction(actionWinSparkle)

        geometry = CONFIG['session/geometry']
        if geometry is None:
            desktop = QtGui.QDesktopWidget()
            screen = desktop.availableGeometry(desktop.primaryScreen())
            self.setFixedSize(screen.width()*0.8, screen.height()*0.9)
        else:
            self.setGeometry(geometry)

        fs = int(1.3 * QtGui.QFont().pointSize())
        self.tabs.setStyleSheet("QTabBar {{ font-size: {}pt; }}".format(fs))
        menu_button.setStyleSheet("QPushButton {{ font-size: {}pt; font-weight: bold; }}".format(fs))

        self.config_changed.connect(update_textedit_colors)

        self.show()

    def _update_info_color(self):
        pal = self.info_table.palette()
        if self.info_model.rowCount() > 0:
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffc"))
        else:
            pal.setColor(QtGui.QPalette.Base, pal.color(QtGui.QPalette.Window))
        self.info_table.setPalette(pal)

    def _on_select_info(self, current, _):
        if not current.isValid(): return
        self.current_controller.select_info(self.info_model.entries[current.row()])
        self.info_table.setCurrentIndex(QtCore.QModelIndex())

    @property
    def current_controller(self):
        if self.current_tab_index == -1: return None
        return self.document.controller_by_index(self.current_tab_index)

    def update_recent_list(self):
        self.recent_menu.clear()
        class Func(object):
            def __init__(s, f): s.f = f
            def __call__(s): return self.open(s.f)
        for i,f in enumerate(reversed(RECENT)):
            action = QtGui.QAction(f, self)
            action.triggered.connect(Func(f))
            # action.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_0 + (i+1)%10))
            self.recent_menu.addAction(action)

    def _try_load_from_file(self, filename):
        update_recent_files(os.path.abspath(filename))
        document = PyDocument(self) if filename.endswith('.py') else XPLDocument(self)
        try:
            document.load_from_file(filename)
        except Exception as e:
            if _DEBUG: raise e
            QtGui.QMessageBox.critical(self, 'Error while loading XPL from file.',
                                       'Error while loading XPL from file "{}":\n{}'.format(filename, str(e)))
            return False
        else:
            self.document = document
            self.setup_model()
            self.set_changed(False)
            return True

    def setup_model(self):
        self.tabs.clear()
        for m in self.document.SECTION_NAMES:
            self.tabs.addTab(self.document.controller_by_name(m).get_widget(), self.SECTION_TITLES[m])
            self.tabs.setTabToolTip(self.tabs.count()-1, self.SECTION_TIPS[m])
            # self.tabs.setTabIcon(self.tabs.count()-1, QtGui.QIcon.fromTheme(SECTION_ICONS[m]))
        self.current_tab_index = -1
        if isinstance(self.document, PyDocument):
            self.tab_change(0)
        else:
            self.tabs.setCurrentIndex(2)

    def new(self):
        new_window = MainWindow()
        new_window.resize(self.size())
        new_window.move(self.x() + 24, self.y() + 24)
        WINDOWS.add(new_window)

    def open(self, filename=None):
        if not filename:
            filename = QtGui.QFileDialog.getOpenFileName(self, "Open file", CURRENT_DIR,
                                                         "PLaSK file (*.xpl *.py);;"
                                                         "PLaSK structure data (*.xpl);;"
                                                         "Python script (*.py)")
            if type(filename) == tuple: filename = filename[0]
            if not filename:
                return
        remove_self = self.document.filename is None and not self.isWindowModified()
        new_window = MainWindow(filename)
        try:
            if new_window.document.filename is not None:
                new_window.resize(self.size())
                WINDOWS.add(new_window)
                if remove_self:
                    self.close()
                else:
                    new_window.move(self.x() + 24, self.y() + 24)
            else:
                new_window.setWindowModified(False)
                new_window.close()
        except AttributeError:
            new_window.setWindowModified(False)
            new_window.close()

    def _save_document(self, filename):
        try:
            fire_edit_end()
            self.document.save_to_file(unicode(filename))
        except Exception as err:
            msgbox = QtGui.QMessageBox()
            msgbox.setWindowTitle("Save Error")
            msgbox.setText("The file '{}' could not be saved to disk.".format(filename))
            msgbox.setInformativeText(unicode(err))
            msgbox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgbox.setIcon(QtGui.QMessageBox.Critical)
            msgbox.exec_()
            return False
        else:
            update_recent_files(os.path.abspath(filename))
            self.setWindowModified(False)
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
        flt = "Python script (*.py)" if isinstance(self.document, PyDocument) else "PLaSK structure data  (*.xpl)"
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save file as", self.document.filename or CURRENT_DIR, flt)
        if type(filename) is tuple:
            filename = filename[0]
        if not filename:
            return False
        return self._save_document(filename)

    def before_save(self):
        """"Is called just before save, return True if document can be saved."""
        if self.current_tab_index != -1:
            try:
                self.document.controller_by_index(self.current_tab_index).save_data_in_model()
            except Exception as e:
                msgbox = QtGui.QMessageBox()
                msgbox.setText("Edited content of the current section is invalid.")
                msgbox.setDetailedText(str(e))
                msgbox.setInformativeText("Do you want to save anyway (with the old content of the current section)?")
                msgbox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                msgbox.setIcon(QtGui.QMessageBox.Warning)
                #msgbox.setDefaultButton(QtGui.QMessageBox.Yes);
                return msgbox.exec_() == QtGui.QMessageBox.Yes
        errors = self.document.get_info(Info.ERROR)
        sys.stdout.flush()
        if errors:
            msgbox = QtGui.QMessageBox()
            msgbox.setText("Document contains some non-critical errors.\n\n"
                           "It is possible to save it, however launching it will most probably fail.")
            msgbox.setDetailedText(u'\n'.join(map(unicode, errors)))
            msgbox.setInformativeText("Do you want to save anyway?")
            msgbox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            msgbox.setIcon(QtGui.QMessageBox.Warning)
            msgbox.setDefaultButton(QtGui.QMessageBox.Yes)
            return msgbox.exec_() == QtGui.QMessageBox.Yes
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
            confirm = QtGui.QMessageBox.question(self, "Unsaved File",
                                                 "File is not saved. Do you want to save it before closing the window?",
                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)
            if confirm == QtGui.QMessageBox.Cancel or (confirm == QtGui.QMessageBox.Yes and not self.save()):
                event.ignore()
                return

        self.closed.emit()

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
            cursor = QtGui.QTextCursor(editor.document().findBlockByLineNumber(
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


class GotoDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        super(GotoDialog, self).__init__(parent)
        self.setWindowTitle("Go to Line")
        vbox = QtGui.QVBoxLayout()
        hbox = QtGui.QHBoxLayout()
        label = QtGui.QLabel()
        label.setText("Line number:")
        self.input = QtGui.QLineEdit()
        self.input.setValidator(QtGui.QIntValidator(self.input))
        hbox.addWidget(label)
        hbox.addWidget(self.input)
        vbox.addLayout(hbox)
        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)
        self.setLayout(vbox)


class PlaskApplication(QtGui.QApplication):

    def __init__(self, argv):
        self._opened_windows = []
        super(PlaskApplication, self).__init__(argv)

    def commitData(self, session_manager):
        self._opened_windows = WINDOWS.copy()
        super(PlaskApplication, self).commitData(session_manager)

    def saveState(self, session_manager):
        files = []
        for window in self._opened_windows:
            if window.document.filename is not None:
                files.append(window.document.filename)
        if files:
            CONFIG['session/saved_' + session_manager.sessionKey()] = files
            CONFIG.sync()

    def restoreState(self):
        key = 'session/saved_' + self.sessionKey()
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
        import traceback
        def excepthook(exc_type, exc_val, tracebackobj):
            msg = ''.join(traceback.format_exception(exc_type, exc_val, tracebackobj)) + '\n'
            sys.stderr.write(msg)
            sys.stderr.flush()
        sys.excepthook = excepthook

    global APPLICATION
    if winsparkle:
        si = subprocess.STARTUPINFO()
        si.dwFlags = subprocess.STARTF_USESTDHANDLES | subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        try:
            ver = plask.version
        except NameError:
            proc = subprocess.Popen(['plask', '-V'], startupinfo=si, stdout=subprocess.PIPE)
            version, err = proc.communicate()
            prog, ver = version.strip().split()
        wp = ctypes.c_wchar_p
        winsparkle.win_sparkle_set_app_details(wp("PLaSK"), wp("PLaSK"), wp(ver))
        winsparkle.win_sparkle_set_appcast_url("http://phys.p.lodz.pl/appcast/plask.xml")
        winsparkle.win_sparkle_set_registry_path("Software\\plask\\updates")
        winsparkle.win_sparkle_init()

    APPLICATION = PlaskApplication(sys.argv)
    APPLICATION.setApplicationName("PLaSK")
    sys.argv = APPLICATION.arguments()

    icons_theme = str(CONFIG['main_window/icons_theme']).lower()
    if icons_theme == 'system':
        icons_path = QtGui.QIcon.themeSearchPaths()[:-1]
        if not QtGui.QIcon.themeName():
            QtGui.QIcon.setThemeName('hicolor')
    else:
        if icons_theme == 'tango': icons_theme = 'hicolor'
        icons_path = []
        QtGui.QIcon.setThemeName(icons_theme)
    icons_path.insert(0, os.path.join(__path__[0], 'icons'))
    QtGui.QIcon.setThemeSearchPaths(icons_path)

    plugins_dir = os.path.join(__path__[0], 'plugins')
    for loader, modname, ispkg in pkgutil.walk_packages([plugins_dir]):
        loader.find_module(modname).load_module(modname)

    if matplotlib:
        ft = QtGui.QWidget().font()
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

    if APPLICATION.isSessionRestored():
        if not APPLICATION.restoreState():
            WINDOWS.add(MainWindow())
    else:
        if len(sys.argv) > 1:
            filename = os.path.abspath(sys.argv[1])
            WINDOWS.add(MainWindow(filename))
        else:
            WINDOWS.add(MainWindow())

    exit_code = APPLICATION.exec_()

    if winsparkle:
        winsparkle.win_sparkle_cleanup()

    sys.exit(exit_code)
