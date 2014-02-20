from PyQt4 import QtGui
from controler.base import Controler

class TableActions(object):
    
    def __init__(self, table, model = None):
        object.__init__(self)
        self.table = table
        self.model = model if model != None else table.model()
        
    def addEntry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            row = self.model.insert(index.row()+1)
        else:
            row = self.model.insert()
        if row != None: self.table.selectRow(row)
    
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
    
    def get(self, parent):
        self.addAction = QtGui.QAction(QtGui.QIcon.fromTheme('list-add'), '&Add', parent)
        self.addAction.setStatusTip('Add new entry to the list')
        self.addAction.triggered.connect(self.addEntry)
            
        self.removeAction = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove'), '&Remove', parent)
        self.removeAction.setStatusTip('Remove selected entry from the list')
        self.removeAction.triggered.connect(self.removeEntry)
            
        self.moveUpAction = QtGui.QAction(QtGui.QIcon.fromTheme('go-up'), 'Move &up', parent)
        self.moveUpAction.setStatusTip('Change order of entries: move current entry up')
        self.moveUpAction.triggered.connect(self.moveUp)
            
        self.moveDownAction = QtGui.QAction(QtGui.QIcon.fromTheme('go-down'), 'Move &down', parent)
        self.moveDownAction.setStatusTip('Change order of entries: move current entry down')
        self.moveDownAction.triggered.connect(self.moveDown)
            
        return self.addAction, self.removeAction, self.moveUpAction, self.moveDownAction

def tableWithManipulators(table, parent = None, model = None, title = None):
    toolBar = QtGui.QToolBar()
    table.table_manipulators_actions = TableActions(table, model)
    toolBar.addActions(table.table_manipulators_actions.get(parent))

    vbox = QtGui.QVBoxLayout()        
    vbox.addWidget(toolBar)
    vbox.addWidget(table)
            
    external = QtGui.QGroupBox()
    if title != None:
        external.setTitle(title)
        m = external.getContentsMargins()
        external.setContentsMargins(0, m[1], 0, m[3])
    else:
        external.setContentsMargins(0, 0, 0, 0)
    vbox.setContentsMargins(0, 0, 0, 0)
        
    external.setLayout(vbox)
    #if title == None:
    #widget.setContentsMargins(0, 0, 0, 0)
    
    return external


class TableControler(Controler):

    def __init__(self, document, model):
        Controler.__init__(self, document, model)
        self.table = QtGui.QTableView()
        self.table.setModel(self.model)
        self.tableActions = TableActions(self.table)
        
        cols = self.model.columnCount(None) #column widths:
        for c in range(0, cols): self.table.setColumnWidth(c, 200)
        self.table.horizontalHeader().setResizeMode(cols-1, QtGui.QHeaderView.Stretch);
        
    def getEditor(self):
        return self.table

    def onEditEnter(self):
        self.saveDataInModel()  #this should do nothing, but is called in case of subclass use it
        if not self.model.isReadOnly():
            self.document.mainWindow.setSectionActions(*self.getTableEditActions())

    # when editor is turn off, model should be update
    def onEditExit(self):
        self.document.mainWindow.setSectionActions()
    
    def getTableEditActions(self):
        return self.tableActions.get(self.document.mainWindow)
