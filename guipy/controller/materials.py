from PyQt4 import QtGui
from controller.base import Controller
from PyQt4.QtGui import QSplitter
from model import materials
from model.materials import MaterialPropertyModel
from utils.gui import HTMLDelegate, table_last_col_fill
from controller.defines import DefinesCompletionDelegate
from controller.table import tableWithManipulators

class MaterialBaseDelegate(DefinesCompletionDelegate):
    
    def __init__(self, definesModel, parent):
        DefinesCompletionDelegate.__init__(self, definesModel, parent)
        
    def createEditor(self, parent, option, index):
        earlier_names = [ e.name for e in index.model().entries[0:index.row()] ]

        if not earlier_names: return super(MaterialBaseDelegate, self).createEditor(parent, option, index)
        
        combo = QtGui.QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(earlier_names)
        combo.setEditText(index.data())
        combo.setCompleter(self.getDefinesCompleter(parent))
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"), 
        #             self, QtCore.SLOT("currentIndexChanged()"))
        return combo
    

class MaterialPropertiesDelegate(DefinesCompletionDelegate):

    def __init__(self, definesModel, parent):
        DefinesCompletionDelegate.__init__(self, definesModel, parent)
        
    def createEditor(self, parent, option, index):
        opts = index.model().options_to_choose(index)

        if opts == None: return super(MaterialPropertiesDelegate, self).createEditor(parent, option, index)
        
        combo = QtGui.QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(opts)
        combo.setEditText(index.data())
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"), 
        #             self, QtCore.SLOT("currentIndexChanged()"))
        return combo
    

class MaterialsController(Controller):

    def __init__(self, document, model = materials.MaterialsModel()):
        Controller.__init__(self, document, model)
        
        self.splitter = QSplitter()
        
        self.materials_table = QtGui.QTableView()
        self.materials_table.setModel(self.model)
        self.materials_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model, self.materials_table))
        #self.materialsTableActions = TableActions(self.materials_table)
        table_last_col_fill(self.materials_table, self.model.columnCount(None), 150)
        self.splitter.addWidget(tableWithManipulators(self.materials_table, self.splitter, title="Materials"))
        
        self.property_model = MaterialPropertyModel(model)
        self.properties_table = QtGui.QTableView()
        self.properties_table.setModel(self.property_model)
        self.properties_delegate = MaterialPropertiesDelegate(self.document.defines.model, self.properties_table)
        self.properties_table.setItemDelegateForColumn(0, self.properties_delegate)
        self.properties_table.setItemDelegateForColumn(1, self.properties_delegate)       
        self.properties_table.setItemDelegateForColumn(2, HTMLDelegate())
        #self.properties_table.setWordWrap(True)        
        table_last_col_fill(self.properties_table, self.property_model.columnCount(None), [100, 150])
        self.properties_table.verticalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.splitter.addWidget(tableWithManipulators(self.properties_table, self.splitter, title="Properties of the material"))
        
        self.materials_table.selectionModel().selectionChanged.connect(self.material_selected) #currentChanged ??
        
    def material_selected(self, newSelection, oldSelection):
        indexes = newSelection.indexes()
        if indexes:
            self.property_model.material = self.model.entries[indexes[0].row()]
        else:
            self.property_model.material = None
        #self.properties_table.resizeColumnsToContents()
        self.properties_table.resizeRowsToContents()
        
    def get_editor(self):
        return self.splitter

    #def onEditEnter(self):
    #    self.saveDataInModel()  #this should do nothing, but is called in case of subclass use it
    #    if not self.model.isReadOnly():
    #        self.document.mainWindow.setSectionActions(*self.get_table_edit_actions())

    # when editor is turn off, model should be update
    #def onEditExit(self):
    #    self.document.mainWindow.setSectionActions()
    
    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.mainWindow)