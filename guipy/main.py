#!/usr/bin/python
# -*- coding: utf-8 -*-

# use new v2 API, Python types instead of Qt 
import sip
for n in ["QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"]: sip.setapi(n, 2)

import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import SIGNAL
from XPLDocument import XPLDocument
from utils import exceptionToMsg
from model.info import InfoListModel, Info

class MainWindow(QtGui.QMainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__()        
        self.document = XPLDocument(self)
        self.current_tab_index = -1
        self.fileName = None
        self.initUI()
        
    def modelIsNew(self):
        self.tabs.clear()
        for m in XPLDocument.SECTION_NAMES:
            self.tabs.addTab(self.document.getControlerByName(m).getEditor(), m)
        self.current_tab_index = 0
        self.currentSectionEnter()
        
    def setNewModel(self, model):
        self.document = model
        self.modelIsNew()
        
    def new(self):
        reply = QtGui.QMessageBox.question(self, "Save", "Save current project?", QtGui.QMessageBox.Yes|QtGui.QMessageBox.No|QtGui.QMessageBox.Cancel)
        if reply == QtGui.QMessageBox.Cancel or (reply == QtGui.QMessageBox.Yes and not self.save()):
            return
        self.fileName = None
        self.setNewModel(XPLDocument(self))
        
    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Choose the name of experiment file to open", ".", "XPL (*.xpl)");
        if not fileName: return;
        self.document.loadFromFile(fileName)
        
    def save(self):
        if self.fileName != None:
            if not self.beforeSave(): return False
            self.document.saveToFile(self.fileName)
            return True
        else:
            return self.saveAs()
        
    def saveAs(self):
        """Ask for filename and save to chosen file. Return true only when file has been saved."""
        if not self.beforeSave(): return False
        fileName = QtGui.QFileDialog.getSaveFileName(self, "Choose the name of experiment file to save", ".", "XPL (*.xpl)");
        if not fileName: return False
        self.fileName = fileName
        self.document.saveToFile(fileName)
        return True
    
    def beforeSave(self):
        """"Is called just before save, return True if document can be saved."""
        if self.current_tab_index != -1:
            try:
                self.document.getControlerByIndex(self.current_tab_index).saveDataInModel()
            except Exception as e:
                msgBox = QtGui.QMessageBox()
                msgBox.setText("Edited content of the current section is invalid.")
                msgBox.setDetailedText(str(e))
                msgBox.setInformativeText("Do you want to save anyway (with the old content of the current section)?")
                msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                msgBox.setIcon(QtGui.QMessageBox.Warning)
                #msgBox.setDefaultButton(QtGui.QMessageBox.Yes);
                return msgBox.exec_() == QtGui.QMessageBox.Yes
        errors = self.document.getInfo(Info.ERROR)
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
    
    def currentSectionExit(self):
        """"Should be called just before left the current section."""
        if self.current_tab_index != -1:
            if not exceptionToMsg(lambda: self.document.getControlerByIndex(self.current_tab_index).onEditExit(),
                                  self.tabs, 'Error while trying to store data from editor'):
                self.tabs.setCurrentIndex(self.current_tab_index)
                return False
        return True
    
    def currentSectionEnter(self):
        """"Should be called just after set the current section."""
        if self.current_tab_index != -1:
            c = self.document.getControlerByIndex(self.current_tab_index)
            self.info_model.setModel(c.model)
            c.onEditEnter()
        else:
            self.info_model.setModel(None)
        
    def tabChange(self, index):
        if index == self.current_tab_index: return
        if not self.currentSectionExit():
            self.tabs.setCurrentIndex(self.current_tab_index)
            return
        self.current_tab_index = index
        self.currentSectionEnter()
        
    def setActions(self, toolbar_name, *actions):
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
            
    def setSectionActions(self, *actions):
        self.setActions('Section edit', *actions)
        
    def setEditorSelectActions(self, *actions):
        self.setActions('Section editor', *actions)
        
    def initUI(self):
        
        # icons: http://standards.freedesktop.org/icon-naming-spec/icon-naming-spec-latest.html
        
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
        saveAsAction.triggered.connect(self.saveAs)

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
    
        self.tabs.connect(self.tabs, SIGNAL("currentChanged(int)"), self.tabChange)
        self.setCentralWidget(self.tabs)
        
        self.info_dock = QtGui.QDockWidget("Warnings", self)
        self.info_dock.setFeatures(QtGui.QDockWidget.NoDockWidgetFeatures)
        self.info_dock.setTitleBarWidget(QtGui.QWidget())
        self.info_model = InfoListModel(None)
        #self.info_table = QtGui.QTableView(self.info_dock)
        self.info_table = QtGui.QListView(self.info_dock)
        self.info_table.setModel(self.info_model)
        #self.info_table.horizontalHeader().hide()
        #self.info_table.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch);
        self.info_dock.setWidget(self.info_table)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.info_dock)
        
        self.info_model.layoutChanged.connect(lambda: self.info_dock.setVisible(self.info_model.rowCount() > 0))
        
        #viewMenu.addAction(self.info_dock.toggleViewAction());
        
        self.setGeometry(200, 200, 550, 450)
        self.setWindowTitle('Main window')  
        
        self.document.loadFromFile('test.xpl')
        self.modelIsNew()
          
        self.show()
        
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()