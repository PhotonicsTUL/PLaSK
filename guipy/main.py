#!/usr/bin/python
# -*- coding: utf-8 -*-

# use new v2 API, Python types instead of Qt 
import sip
for n in ["QDate", "QDateTime", "QString", "QTextStream", "QTime", "QUrl", "QVariant"]: sip.setapi(n, 2)

import sys
from PyQt4 import QtGui
from PyQt4.QtCore import SIGNAL
from XPLDocument import XPLDocument
from utils import exceptionToMsg

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
        self.setNewModel(XPLDocument())
        self.fileName = None
        
    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Choose the name of experiment file to open", ".", "XPL (*.xpl)");
        if not fileName: return;
        self.document.loadFromFile(fileName)
        
    def save(self):
        if self.fileName != None:
            self.currentSectionExit()
            self.document.saveToFile(self.fileName)
            return True
        else:
            return self.saveAs()
        
    def saveAs(self):
        self.currentSectionExit()
        fileName = QtGui.QFileDialog.getSaveFileName(self, "Choose the name of experiment file to save", ".", "XPL (*.xpl)");
        if not fileName: return False
        self.fileName = fileName
        self.document.saveToFile(fileName)
        return True
    
    def currentSectionExit(self):
        """"Should be called just before left the current section."""
        if self.current_tab_index != -1:
            if not exceptionToMsg(lambda: self.document.getControlerByIndex(self.current_tab_index).onEditExit(self),
                                  self.tabs, 'Error while trying to store data from editor'):
                self.tabs.setCurrentIndex(self.current_tab_index)
                return False
        return True
    
    def currentSectionEnter(self):
        """"Should be called just after set the current section."""
        if self.current_tab_index != -1:
            self.document.getControlerByIndex(self.current_tab_index).onEditEnter(self)
        
    def tabChange(self, index):
        if index == self.current_tab_index: return
        if not self.currentSectionExit():
            self.tabs.setCurrentIndex(self.current_tab_index)
            return
        self.current_tab_index = index
        self.currentSectionEnter()
            
    def setSectionActions(self, *actions):
        self.section_toolbar.clear()
        for a in actions:
            if not a:
                self.section_toolbar.addSeparator()
            else:
                self.section_toolbar.addAction(a)
        self.section_toolbar.setVisible(bool(actions))
        
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
        
        self.section_toolbar = self.addToolBar('Section')
        self.section_toolbar.hide()
        
        self.tabs = QtGui.QTabWidget(self)
        self.tabs.setDocumentMode(True)
    
        self.tabs.connect(self.tabs, SIGNAL("currentChanged(int)"), self.tabChange)
        self.setCentralWidget(self.tabs)
        
        self.setGeometry(200, 200, 550, 450)
        self.setWindowTitle('Main window')  
        
        self.document.loadFromFile('../doc/tutorial1.xpl')
        self.modelIsNew()
          
        self.show()
        
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()