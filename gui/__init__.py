#!/usr/bin/python
# -*- coding: utf-8 -*-

# use new v2 API, Python types instead of Qt
import sys
import os
import ctypes
import subprocess
import pkgutil

from .qt import QtGui, QtCore

from . import _resources

from .XPLDocument import XPLDocument
from .utils.gui import exception_to_msg
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


class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.document = XPLDocument(self)
        self.current_tab_index = -1
        self.filename = None
        self.init_ui()

    def make_window_title(self):
        if self.filename:
            self.setWindowTitle("{} - PLaSK".format(self.filename))
        else:
            self.setWindowTitle("PLaSK")

    def model_is_new(self):
        self.tabs.clear()
        for m in XPLDocument.SECTION_NAMES:
            self.tabs.addTab(self.document.controller_by_name(m).get_editor(), m.title())
        self.current_tab_index = 0
        self.current_section_enter()

    def set_model(self, model):
        self.document = model
        self.model_is_new()

    def new(self):
        reply = QtGui.QMessageBox.question(self, "Save", "Save current project?",
                                           QtGui.QMessageBox.Yes|QtGui.QMessageBox.No|QtGui.QMessageBox.Cancel)
        if reply == QtGui.QMessageBox.Cancel or (reply == QtGui.QMessageBox.Yes and not self.save()):
            return
        self.filename = None
        self.make_window_title()
        self.set_model(XPLDocument(self))

    def open(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open file", "", "PLaSK structure data (*.xpl)")
        if not filename: return;
        if type(filename) == tuple:
            filename = filename[0]
        self.filename = filename
        self.document.load_from_file(filename)
        self.make_window_title()

    def save(self):
        if self.filename != None:
            if not self.before_save(): return False
            self.document.save_to_file(self.filename)
            return True
        else:
            return self.save_as()

    def save_as(self):
        """Ask for filename and save to chosen file. Return true only when file has been saved."""
        if not self.before_save(): return False
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save file as", self.filename or "",
                                                     "PLaSK structure data  (*.xpl)")
        if type(filename) is tuple: filename = filename[0]
        if not filename: return False
        self.filename = filename
        self.make_window_title()
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
            msgbox.setText("Document contains some non-critical errors. It is possible to save it but probably not to run.")
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

    def set_actions(self, toolbar_name, *actions):
        try:
            toolbar = self.toolbars[toolbar_name]
        except KeyError:
            toolbar = self.addToolBar(toolbar_name)
            self.toolbars[toolbar_name] = toolbar
        toolbar.clear()
        for a in actions:
            if not a:
                toolbar.addSeparator()
            else:
                toolbar.addAction(a)
        toolbar.setVisible(bool(actions))

    def set_section_actions(self, *actions):
        self.set_actions('Section edit', *actions)

    def set_editor_select_actions(self, *actions):
        self.set_actions('Section editor', *actions)

    def init_ui(self):

        # icons: http://standards.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
        icon = QtGui.QIcon(':/plask.png')
        self.setWindowIcon(icon)

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

        self.menubar = self.menuBar()
        self.file_menu = self.menubar.addMenu('&File')
        self.file_menu.addAction(new_action)
        self.file_menu.addAction(open_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(save_action)
        self.file_menu.addAction(saveas_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(launch_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(exit_action)

        #view_menu = menubar.addMenu('&View')

        #edit_menu = menubar.addMenu('&Edit')
        #edit_menu.addAction(showSourceAction)

        global winsparkle
        if winsparkle:
            tools_menu = self.menubar.addMenu('&Tools')
            if winsparkle:
                tools_menu.addSeparator()
                try:
                    actionWinSparkleAutoupdate = QtGui.QAction(self)
                    actionWinSparkleAutoupdate.setText(self.tr("Automatic Updates"))
                    actionWinSparkleAutoupdate.setCheckable(True)
                    actionWinSparkleAutoupdate.setChecked(winsparkle.win_sparkle_get_automatic_check_for_updates())
                    actionWinSparkleAutoupdate.triggered.connect(
                        lambda: winsparkle.win_sparkle_set_automatic_check_for_updates(
                            int(actionWinSparkleAutoupdate.isChecked())))
                    tools_menu.addAction(actionWinSparkleAutoupdate)
                except AttributeError:
                    pass
                actionWinSparkle = QtGui.QAction(self)
                actionWinSparkle.setText("Check for Updates Now...")
                actionWinSparkle.triggered.connect(lambda: winsparkle.win_sparkle_check_update_with_ui())
                tools_menu.addAction(actionWinSparkle)

        toolbar = self.addToolBar('File')
        toolbar.addAction(new_action)
        toolbar.addAction(open_action)
        toolbar.addAction(save_action)
        toolbar.addAction(saveas_action)
        toolbar.addSeparator()
        toolbar.addAction(launch_action)
        #toolbar.addAction(exit_action)
        #toolbar.addSeparator()
        #toolbar.addAction(showSourceAction)
        self.toolbars = {}
        self.toolbars['File'] = toolbar

        self.tabs = QtGui.QTabWidget(self)
        self.tabs.setDocumentMode(True)

        self.tabs.connect(self.tabs, QtCore.SIGNAL("currentChanged(int)"), self.tab_change)
        self.setCentralWidget(self.tabs)

        self.info_dock = QtGui.QDockWidget("Warnings", self)
        self.info_dock.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self.info_dock.setTitleBarWidget(QtGui.QWidget())
        self.info_model = InfoTreeModel(None)
        #self.info_table = QtGui.QTableView(self.info_dock)
        self.info_table = QtGui.QListView(self.info_dock)
        self.info_table.setModel(self.info_model)
        #self.info_table.horizontalHeader().hide()
        #self.info_table.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch);
        self.info_dock.setWidget(self.info_table)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.info_dock)

        self.info_model.layoutChanged.connect(lambda: self.info_dock.setVisible(self.info_model.rowCount() > 0))

        #viewMenu.addAction(self.info_dock.toggleViewAction());

        geometry = CONFIG['session/geometry']
        if geometry is None:
            desktop = QtGui.QDesktopWidget()
            screen = desktop.availableGeometry(desktop.primaryScreen())
            self.setFixedSize(screen.width()*0.8, screen.height()*0.9)
        else:
            self.setGeometry(geometry)

        self.make_window_title()

        if len(sys.argv) > 1:
            try:
                self.document.load_from_file(os.path.join(os.path.dirname(__file__), sys.argv[1]))
            except IOError:
                pass
            else:
                self.filename = sys.argv[1]
                self.make_window_title()
        self.model_is_new()

        self.show()

    def quitting(self):
        geometry = self.geometry()
        CONFIG['session/geometry'] = geometry
        CONFIG.sync()


def main():
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

    app = QtGui.QApplication(sys.argv)
    main_window = MainWindow()
    app.aboutToQuit.connect(main_window.quitting)

    plugins_dir = os.path.join(__path__[0], 'plugins')
    for loader, modname, ispkg in pkgutil.walk_packages([plugins_dir]):
        mod = loader.find_module(modname).load_module(modname)

    exit_code = app.exec_()

    if winsparkle:
        winsparkle.win_sparkle_cleanup()

    sys.exit(exit_code)
