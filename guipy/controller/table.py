from PyQt4 import QtGui
from controller.base import Controller

class TableActions(object):
    
    def __init__(self, table, model = None):
        object.__init__(self)
        self.table = table
        self.model = model if model != None else table.model()
        
    def add_entry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            row = self.model.insert(index.row()+1)
        else:
            row = self.model.insert()
        if row != None: self.table.selectRow(row)
    
    def remove_entry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            self.model.remove(index.row())
    
    def move_up(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 1 <= index < len(self.model.entries):
            self.model.swapNeighbourEntries(index-1, index)
            #self.table.selectRow(index-1)
    
    def move_down(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 0 <= index < len(self.model.entries)-1:
            self.model.swapNeighbourEntries(index, index+1)
            #self.table.selectRow(index+1)
    
    def get(self, parent):
        self.addAction = QtGui.QAction(QtGui.QIcon.fromTheme('list-add'), '&Add', parent)
        self.addAction.setStatusTip('Add new entry to the list')
        self.addAction.triggered.connect(self.add_entry)
            
        self.removeAction = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove'), '&Remove', parent)
        self.removeAction.setStatusTip('Remove selected entry from the list')
        self.removeAction.triggered.connect(self.remove_entry)
            
        self.moveUpAction = QtGui.QAction(QtGui.QIcon.fromTheme('go-up'), 'Move &up', parent)
        self.moveUpAction.setStatusTip('Change order of entries: move current entry up')
        self.moveUpAction.triggered.connect(self.move_up)
            
        self.moveDownAction = QtGui.QAction(QtGui.QIcon.fromTheme('go-down'), 'Move &down', parent)
        self.moveDownAction.setStatusTip('Change order of entries: move current entry down')
        self.moveDownAction.triggered.connect(self.move_down)
            
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


class TableController(Controller):

    def __init__(self, document, model):
        Controller.__init__(self, document, model)
        self.table = QtGui.QTableView()
        self.table.setModel(self.model)
        self.tableActions = TableActions(self.table)
        
        cols = self.model.columnCount(None) #column widths:
        for c in range(0, cols): self.table.setColumnWidth(c, 200)
        self.table.horizontalHeader().setResizeMode(cols-1, QtGui.QHeaderView.Stretch);
        
    def get_editor(self):
        return self.table

    def on_edit_enter(self):
        self.save_data_in_model()  #this should do nothing, but is called in case of subclass use it
        if not self.model.isReadOnly():
            self.document.mainWindow.setSectionActions(*self.get_table_edit_actions())

    # when editor is turn off, model should be update
    def on_edit_exit(self):
        self.document.mainWindow.setSectionActions()
    
    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.mainWindow)
