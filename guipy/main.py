#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt4 import QtGui
from PyQt4.QtCore import SIGNAL
from model import Model

class MainWindow(QtGui.QMainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__()        
        self.model = Model()
        self.current_tab_index = -1
        self.fileName = None
        self.initUI()
        
    def modelIsNew(self):
        self.tabs.clear()
        for m in Model.NAMES:
            self.tabs.addTab(self.model.getModelByName(m).getEditor(), m)
        
    def setNewModel(self, model):
        self.model = model
        self.modelIsNew()
        
    def new(self):
        reply = QtGui.QMessageBox.question(self, "Save", "Save current project?", QtGui.QMessageBox.Yes|QtGui.QMessageBox.No|QtGui.QMessageBox.Cancel)
        if reply == QtGui.QMessageBox.Cancel or (reply == QtGui.QMessageBox.Yes and not self.save()):
            return
        self.setNewModel(Model())
        self.fileName = None
        
    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Choose the name of experiment file to open", ".", "XPL (*.xpl)");
        if fileName.isEmpty(): return;
        self.model.loadFromFile(fileName)
        
    def updateModel(self):
        if self.current_tab_index != -1:
            self.model.getModelByIndex(self.current_tab_index).afterEdit(self.tabs.currentWidget()) # currentWidget is current editor, or prev. one?
        
    def save(self):
        if self.fileName != None:
            self.model.saveToFile(self.fileName)
            return True
        else:
            return self.saveAs()
        
    def saveAs(self):
        self.updateModel()
        fileName = QtGui.QFileDialog.getSaveFileName(self, "Choose the name of experiment file to save", ".", "XPL (*.xpl)");
        if fileName.isEmpty(): return False
        self.fileName = fileName
        self.model.saveToFile(fileName)
        return True
        
    def tabChange(self, index):
        if index == self.current_tab_index: return
        self.updateModel()
        self.current_tab_index = index
        
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
        
        showSourceAction = QtGui.QAction(QtGui.QIcon.fromTheme('accessories-text-editor'), '&Show source', self)
        showSourceAction.setCheckable(True)
        showSourceAction.setStatusTip('Show XPL source of the current section')
        #newAction.triggered.connect(self.new)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addSeparator()
        fileMenu.addAction(saveAction)
        fileMenu.addAction(saveAsAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)
        
        editMenu = menubar.addMenu('&Edit')
        editMenu.addAction(showSourceAction)

        toolbar = self.addToolBar('File')
        toolbar.addAction(newAction)
        toolbar.addAction(openAction)
        toolbar.addAction(saveAction)
        toolbar.addAction(saveAsAction)
        toolbar.addAction(exitAction)
        toolbar.addSeparator()
        toolbar.addAction(showSourceAction)
        
        self.tabs = QtGui.QTabWidget(self)
        self.tabs.setDocumentMode(True)
    
        self.tabs.connect(self.tabs, SIGNAL("currentChanged(int)"), self.tabChange)
        self.setCentralWidget(self.tabs)
        
        self.setGeometry(200, 200, 550, 450)
        self.setWindowTitle('Main window')  
        
        self.model.loadFromFile('/home/qwak/plask/trunk/doc/tutorial1.xpl')
        self.modelIsNew()
          
        self.show()
        
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()