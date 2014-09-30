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

_DEBUG = False

import sys
import os
import ctypes
import subprocess
import pkgutil

from .qt import QtGui, QtCore, qt

# Set-up corrent backend for matplotlib
try: import matplotlib
except ImportError: pass
else: matplotlib.rc('backend', qt4=qt)

from . import _resources

from .xpldocument import XPLDocument
from .pydocument import PyDocument
from .material_plot import show_material_plot
from .utils.widgets import exception_to_msg
from .model.info import InfoTreeModel, Info
from .launch import launch_plask

from .utils.config import CONFIG

try:
    import plask
except ImportError:
    pass

try:
    winsparkle = ctypes.CDLL('WinSparkle.dll')
except OSError:
    winsparkle = None

WINDOWS = set()

MENU_ITEMS = []

class MainWindow(QtGui.QMainWindow):

    def __init__(self, filename=None):
        super(MainWindow, self).__init__()
        self.current_tab_index = -1

        self.tabs = QtGui.QTabWidget(self)
        self.tabs.setDocumentMode(True)

        self.setCentralWidget(self.tabs)

        self.toolbars = {}

        self.showsource_action = QtGui.QAction(
            QtGui.QIcon.fromTheme('accessories-text-editor', QtGui.QIcon(':/accessories-text-editor.png')),
            '&Show source', self)
        self.showsource_action.setCheckable(True)
        self.showsource_action.setStatusTip('Show XPL source of the current section')
        self.showsource_action.setEnabled(False)

        # icons: http://standards.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
        icon = QtGui.QIcon(':/plask.png')
        self.setWindowIcon(icon)

        self.info_dock = QtGui.QDockWidget("Warnings", self)
        self.info_dock.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self.info_dock.setTitleBarWidget(QtGui.QWidget())
        self.info_model = InfoTreeModel(None)
        #self.info = QtGui.QTableView(self.plot_dock)
        self.info_table = QtGui.QListView(self.info_dock)
        self.info_table.setModel(self.info_model)
        self.info_table.setSelectionMode(QtGui.QListView.NoSelection)
        pal = self.info_table.palette()
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffc"))
        self.info_table.setPalette(pal)
        #self.info.horizontalHeader().hide()
        #self.info.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch);
        self.info_dock.setWidget(self.info_table)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.info_dock)

        self.info_model.layoutChanged.connect(lambda: self.info_dock.setVisible(self.info_model.rowCount() > 0))

        self.material_plot_action = QtGui.QAction(QtGui.QIcon.fromTheme('edit-find', QtGui.QIcon(':/edit-find.png')),
                                                  '&Material parameters...', self)
        self.material_plot_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_M)
        self.material_plot_action.triggered.connect(lambda: show_material_plot(self, self.document.materials.model))

        if filename is not None:
            self._try_load_from_file(filename)
        else:
            self.document = XPLDocument(self)
            self.model_is_new()

        #self.statusBar()

        new_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-new', QtGui.QIcon(':/document-new.png')),
                                   '&New', self)
        new_action.setShortcut(QtGui.QKeySequence.New)
        new_action.setStatusTip('New XPL file')
        new_action.triggered.connect(self.new)

        open_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-open', QtGui.QIcon(':/document-open.png')),
                                    '&Open...', self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.setStatusTip('Open XPL file')
        open_action.triggered.connect(self.open)

        save_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-save', QtGui.QIcon(':/document-save.png')),
                                    '&Save', self)
        save_action.setShortcut(QtGui.QKeySequence.Save)
        save_action.setStatusTip('Save XPL file')
        save_action.triggered.connect(self.save)

        saveas_action = QtGui.QAction(QtGui.QIcon.fromTheme('document-save-as', QtGui.QIcon(':/document-save-as.png')),
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

        exit_action = QtGui.QAction(QtGui.QIcon.fromTheme('exit', QtGui.QIcon(':/exit.png')), 'E&xit', self)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)

        self.menu = QtGui.QMenu('&Operations')
        self.menu.addAction(new_action)
        self.menu.addAction(open_action)
        self.menu.addSeparator()
        self.menu.addAction(save_action)
        self.menu.addAction(saveas_action)
        self.menu.addSeparator()
        self.menu.addAction(launch_action)
        self.menu.addSeparator()
        self.menu.addAction(self.material_plot_action)

        self.menu.addSeparator()
        self.menu.addAction(exit_action)

        menu_button = QtGui.QPushButton(self)
        menu_button.setText("&Operations")
        pal = menu_button.palette()
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#88aaff"))
        menu_button.setPalette(pal)

        menu_button.setMenu(self.menu)
        self.tabs.setCornerWidget(menu_button, QtCore.Qt.TopLeftCorner)

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

        self.tabs.currentChanged[int].connect(self.tab_change)

        self.show()

    def _try_load_from_file(self, filename):
        if filename.endswith('.py'):
            document = PyDocument(self)
        else:
            document = XPLDocument(self)
        try:
            document.load_from_file(filename)
        except Exception as e:
            if _DEBUG: raise
            QtGui.QMessageBox.critical(self, 'Error while loading XPL from file.',
                                       'Error while loading XPL from file "{}":\n{}'.format(filename, str(e)))
            return False
        else:
            self.set_model(document)
            return True

    def model_is_new(self):
        self.tabs.clear()
        for m in XPLDocument.SECTION_NAMES:
            self.tabs.addTab(self.document.controller_by_name(m).get_editor(), m.title())
        self.current_tab_index = 0
        new_index = 2
        self.tabs.setCurrentIndex(new_index)
        self.tab_change(new_index)

    def set_model(self, model):
        self.document = model
        self.model_is_new()

    def new(self):
        new_window = MainWindow()
        new_window.resize(self.size())
        WINDOWS.add(new_window)

    def open(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open file", "",
                                                     "PLaSK file (*.xpl *.py);;"
                                                     "PLaSK structure data (*.xpl);;"
                                                     "Python script (*.py)")
        if not filename: return;
        if type(filename) == tuple: filename = filename[0]
        if self.document.filename is None and not self.isWindowModified():
            self._try_load_from_file(filename)
        else:
            new_window = MainWindow(filename)
            if new_window.document.filename is not None:
                new_window.resize(self.size())
                WINDOWS.add(new_window)
            else:
                new_window.close()

    def save(self):
        if self.document.filename is not None:
            if not self.before_save(): return False
            self.document.save_to_file(self.document.filename)
            return True
        else:
            return self.save_as()

    def save_as(self):
        """Ask for filename and save to chosen file. Return true only when file has been saved."""
        if not self.before_save(): return False
        filter = "Python script (*.py)" if isinstance(self.document, PyDocument) else "PLaSK structure data  (*.xpl)"
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save file as", self.document.filename or "", filter)
        if type(filename) is tuple: filename = filename[0]
        if not filename: return False
        self.document.save_to_file(filename)
        return True

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
        if errors:
            msgbox = QtGui.QMessageBox()
            msgbox.setText("Document contains some non-critical errors. "
                           "It is possible to save it but probably not to run.")
            msgbox.setDetailedText('\n'.join(map(str, errors)))
            msgbox.setInformativeText("Do you want to save anyway?")
            msgbox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            msgbox.setIcon(QtGui.QMessageBox.Warning)
            msgbox.setDefaultButton(QtGui.QMessageBox.Yes)
            return msgbox.exec_() == QtGui.QMessageBox.Yes
        return True

    def current_section_exit(self):
        """"Should be called just before left the current section."""
        if self.current_tab_index != -1:
            if not exception_to_msg(lambda: self.document.controller_by_index(self.current_tab_index).on_edit_exit(),
                                  self.tabs, 'Error while trying to store data from editor'):
                self.tabs.setCurrentIndex(self.current_tab_index)
                return False
        return True

    def current_section_enter(self):
        """"Should be called just after set the current section."""
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

    def set_actions(self, name, *actions):
        try:
            toolbar = self.toolbars[name]
        except KeyError:
            toolbar = self.addToolBar(name)
            self.toolbars[name] = toolbar
        toolbar.clear()
        for a in actions:
            if not a:
                toolbar.addSeparator()
            else:
                toolbar.addAction(a)
        toolbar.setVisible(bool(actions))

    def set_section_actions(self, *actions):
        self.set_actions('Section', *actions)

    def add_tools_actions(self, actions):
        for i, action in enumerate(actions):
            if not action:
                actions[i] = self.tools_menu.addSeparator()
            else:
                try:
                    self.tools_menu.addAction(action, self.last_tool)
                except AttributeError:
                    self.tools_menu.addAction(action)

    def remove_tools_actions(self, actions):
        for action in actions:
            self.tools_menu.removeAction(action)

    def closeEvent(self, event):
        if self.isWindowModified():
            confirm = QtGui.QMessageBox.question(self, "Unsaved File",
                                                 "File is not saved. Do you want to save it before closing the window?",
                                                 QtGui.QMessageBox.Yes | QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel)
            if confirm == QtGui.QMessageBox.Cancel or (confirm == QtGui.QMessageBox.Yes and not self.save()):
                event.ignore()
                return

        geometry = self.geometry()
        CONFIG['session/geometry'] = geometry
        CONFIG.sync()

    def set_changed(self, changed):
        """Set changed flags in the document window"""
        try:
            if self.document.filename:
                self.setWindowTitle("{}[*] - PLaSK".format(self.document.filename))
            else:
                self.setWindowTitle("[*] PLaSK")
        except AttributeError:
            self.setWindowTitle("[*] PLaSK")
        self.setWindowModified(changed)


def main():
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

    APPLICATION = QtGui.QApplication(sys.argv)
    APPLICATION.setApplicationName("PLaSK")

    plugins_dir = os.path.join(__path__[0], 'plugins')
    for loader, modname, ispkg in pkgutil.walk_packages([plugins_dir]):
        mod = loader.find_module(modname).load_module(modname)

    if len(sys.argv) > 1:
        WINDOWS.add(MainWindow(sys.argv[1]))
    else:
        WINDOWS.add(MainWindow())

    exit_code = APPLICATION.exec_()

    if winsparkle:
        winsparkle.win_sparkle_cleanup()

    sys.exit(exit_code)
