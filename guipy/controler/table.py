from PyQt4 import QtGui
from controler.base import Controler

class TableControler(Controler):

    def __init__(self, document, model):
        Controler.__init__(self, document, model)
        self.table = QtGui.QTableView()
        self.table.setModel(self.model)
        
        cols = self.model.columnCount(None) #column widths:
        for c in range(0, cols): self.table.setColumnWidth(c, 200)
        self.table.horizontalHeader().setResizeMode(cols-1, QtGui.QHeaderView.Stretch);
        
    def getEditor(self):
        return self.table

    def onEditEnter(self):
        self.saveDataInModel()  #this should do nothing, but is called in case of subclass use it
        self.document.mainWindow.setSectionActions(*self.getTableEditActions())

    # when editor is turn off, model should be update
    def onEditExit(self):
        self.document.mainWindow.setSectionActions()
        
    def addEntry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            row = self.model.insert(index.row()+1)
        else:
            row = self.model.insert()
        self.table.selectRow(row)
    
    def removeEntry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            self.model.remove(index.row())
    
    def moveUp(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 1 <= index < len(self.model.entries):
            self.model.swapNeighbourEntries(index-1, index)
            #self.table.selectRow(index-1)
    
    def moveDown(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 0 <= index < len(self.model.entries)-1:
            self.model.swapNeighbourEntries(index, index+1)
            #self.table.selectRow(index+1)
    
    def getTableEditActions(self):
            self.addAction = QtGui.QAction(QtGui.QIcon.fromTheme('list-add'), '&Add', self.document.mainWindow)
            self.addAction.setStatusTip('Add new entry to the list')
            self.addAction.triggered.connect(self.addEntry)
            
            self.removeAction = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove'), '&Remove', self.document.mainWindow)
            self.removeAction.setStatusTip('Remove selected entry from the list')
            self.removeAction.triggered.connect(self.removeEntry)
            
            self.moveUpAction = QtGui.QAction(QtGui.QIcon.fromTheme('go-up'), 'Move &up', self.document.mainWindow)
            self.moveUpAction.setStatusTip('Change order of entries: move current entry up')
            self.moveUpAction.triggered.connect(self.moveUp)
            
            self.moveDownAction = QtGui.QAction(QtGui.QIcon.fromTheme('go-down'), 'Move &down', self.document.mainWindow)
            self.moveDownAction.setStatusTip('Change order of entries: move current entry down')
            self.moveDownAction.triggered.connect(self.moveDown)
            
            return self.addAction, self.removeAction, self.moveUpAction, self.moveDownAction