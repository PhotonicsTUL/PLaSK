from PyQt4 import QtGui
from controler.base import Controler
from PyQt4.QtGui import QSplitter
from model import materials
from model.materials import MaterialPropertyModel, MATERIALS_PROPERTES
from utils import HTMLDelegate, table_last_col_fill, ComboBoxDelegate
from controler.defines import DefinesCompletionDelegate

class MaterialsControler(Controler):

    def __init__(self, document, model = materials.MaterialsModel()):
        Controler.__init__(self, document, model)
        
        self.splitter = QSplitter()
        
        self.materials_table = QtGui.QTableView()
        self.materials_table.setModel(self.model)
        table_last_col_fill(self.materials_table, self.model.columnCount(None), 150)
        self.splitter.addWidget(self.materials_table)
        
        self.property_model = MaterialPropertyModel(model)
        self.properties_table = QtGui.QTableView()
        self.properties_table.setModel(self.property_model)
        self.properties_table.setItemDelegateForColumn(0, ComboBoxDelegate(MATERIALS_PROPERTES.keys(), self.properties_table))
        self.properties_table.setItemDelegateForColumn(1, DefinesCompletionDelegate(self.document.defines.model, self.properties_table))       
        self.properties_table.setItemDelegateForColumn(2, HTMLDelegate())
        #self.properties_table.setWordWrap(True)        
        table_last_col_fill(self.properties_table, self.property_model.columnCount(None), [50, 200])
        self.properties_table.verticalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.splitter.addWidget(self.properties_table)
        
        self.materials_table.selectionModel().selectionChanged.connect(self.material_selected) #currentChanged ??
        
    def material_selected(self, newSelection, oldSelection):
        indexes = newSelection.indexes()
        if len(indexes) >= 1:
            self.property_model.material = self.model.entries[indexes[0].row()]
        else:
            self.property_model.material = None
        #self.properties_table.resizeColumnsToContents()
        self.properties_table.resizeRowsToContents()
        
    def getEditor(self):
        return self.splitter

    def onEditEnter(self):
        self.saveDataInModel()  #this should do nothing, but is called in case of subclass use it
        if not self.model.isReadOnly():
            self.document.mainWindow.setSectionActions(*self.getTableEditActions())

    # when editor is turn off, model should be update
    def onEditExit(self):
        self.document.mainWindow.setSectionActions()
        
    def addEntry(self):
        index = self.materials_table.selectionModel().currentIndex()
        if index.isValid():
            row = self.model.insert(index.row()+1)
        else:
            row = self.model.insert()
        self.materials_table.selectRow(row)
    
    def removeEntry(self):
        index = self.materials_table.selectionModel().currentIndex()
        if index.isValid():
            self.model.remove(index.row())
    
    def moveUp(self):
        index = self.materials_table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 1 <= index < len(self.model.entries):
            self.model.swapNeighbourEntries(index-1, index)
            #self.table.selectRow(index-1)
    
    def moveDown(self):
        index = self.materials_table.selectionModel().currentIndex()
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