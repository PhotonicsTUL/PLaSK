#!/usr/bin/python
# -*- coding: utf-8 -*-

# use new v2 API, Python types instead of Qt
import sip
for n in ("QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"): sip.setapi(n, 2)

import sys
import os
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import SIGNAL

from .resources import APP_ICON
from .XPLDocument import XPLDocument
from .utils.gui import exception_to_msg
from .model.info import InfoTreeModel, Info

from .utils.config import CONFIG

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.document = XPLDocument(self)
        self.current_tab_index = -1
        self.filename = None
        self.init_UI()

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
        self.set_model(XPLDocument(self))

    def open(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open file", "", "XPL (*.xpl)");
        if not filename: return;
        self.document.load_from_file(filename)

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
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save file as", "", "XPL (*.xpl)");
        if not filename: return False
        self.filename = filename
        self.document.save_to_file(filename)
        return True

    def before_save(self):
        """"Is called just before save, return True if document can be saved."""
        if self.current_tab_index != -1:
            try:
                self.document.controller_by_index(self.current_tab_index).save_data_in_model()
            except Exception as e:
                msgBox = QtGui.QMessageBox()
                msgBox.setText("Edited content of the current section is invalid.")
                msgBox.setDetailedText(str(e))
                msgBox.setInformativeText("Do you want to save anyway (with the old content of the current section)?")
                msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                msgBox.setIcon(QtGui.QMessageBox.Warning)
                #msgBox.setDefaultButton(QtGui.QMessageBox.Yes);
                return msgBox.exec_() == QtGui.QMessageBox.Yes
        errors = self.document.get_info(Info.ERROR)
        if errors:
            msgBox = QtGui.QMessageBox()
            msgBox.setText("Document contains some non-critical errors. It is possible to save it but probably not to run.")
            msgBox.setDetailedText('\n'.join(map(str, errors)))
            msgBox.setInformativeText("Do you want to save anyway?")
            msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            msgBox.setIcon(QtGui.QMessageBox.Warning)
            msgBox.setDefaultButton(QtGui.QMessageBox.Yes)
            return msgBox.exec_() == QtGui.QMessageBox.Yes
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
            toolbar = self.extra_toolbars[toolbar_name]
        except KeyError:
            toolbar = self.addToolBar(toolbar_name)
            self.extra_toolbars[toolbar_name] = toolbar
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

    def init_UI(self):

        # icons: http://standards.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
        icon_pixmap = QtGui.QPixmap()
        icon_pixmap.loadFromData(QtCore.QByteArray.fromBase64(APP_ICON))
        icon = QtGui.QIcon()
        icon.addPixmap(icon_pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        #self.statusBar()

        newAction = QtGui.QAction(QtGui.QIcon.fromTheme('document-new'), '&New', self)
        newAction.setShortcut(QtGui.QKeySequence.New)
        newAction.setStatusTip('New XPL file')
        newAction.triggered.connect(self.new)

        openAction = QtGui.QAction(QtGui.QIcon.fromTheme('document-open'), '&Open...', self)
        openAction.setShortcut(QtGui.QKeySequence.Open)
        openAction.setStatusTip('Open XPL file')
        openAction.triggered.connect(self.open)

        saveAction = QtGui.QAction(QtGui.QIcon.fromTheme('document-save'), '&Save', self)
        saveAction.setShortcut(QtGui.QKeySequence.Save)
        saveAction.setStatusTip('Save XPL file')
        saveAction.triggered.connect(self.save)

        saveAsAction = QtGui.QAction(QtGui.QIcon.fromTheme('document-save-as'), 'Save &As...', self)
        saveAsAction.setShortcut(QtGui.QKeySequence.SaveAs)
        saveAsAction.setStatusTip('Save XPL file, ask for name of file')
        saveAsAction.triggered.connect(self.save_as)

        exitAction = QtGui.QAction(QtGui.QIcon.fromTheme('exit'), 'E&xit', self)
        exitAction.setShortcut(QtGui.QKeySequence.Quit)
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addSeparator()
        fileMenu.addAction(saveAction)
        fileMenu.addAction(saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)

        #viewMenu = menubar.addMenu('&View')

        #editMenu = menubar.addMenu('&Edit')
        #editMenu.addAction(showSourceAction)

        toolbar = self.addToolBar('File')
        toolbar.addAction(newAction)
        toolbar.addAction(openAction)
        toolbar.addAction(saveAction)
        toolbar.addAction(saveAsAction)
        #toolbar.addAction(exitAction)
        #toolbar.addSeparator()
        #toolbar.addAction(showSourceAction)
        self.extra_toolbars = {}

        self.tabs = QtGui.QTabWidget(self)
        self.tabs.setDocumentMode(True)

        self.tabs.connect(self.tabs, SIGNAL("currentChanged(int)"), self.tab_change)
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

        self.setWindowTitle('PLaSK')

        if len(sys.argv) > 1:
            try:
                self.document.load_from_file(os.path.join(os.path.dirname(__file__), sys.argv[1]))
            except IOError:
                pass
            else:
                self.filename = sys.argv[1]
        self.model_is_new()

        self.show()

    def quitting(self):
        geometry = self.geometry()
        CONFIG['session/geometry'] = geometry
        CONFIG.sync()


def main():
    app = QtGui.QApplication(sys.argv)
    mw = MainWindow()
    app.aboutToQuit.connect(mw.quitting)
    sys.exit(app.exec_())
