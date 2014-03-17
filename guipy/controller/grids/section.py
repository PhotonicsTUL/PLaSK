from PyQt4 import QtGui
from controller.base import Controller
from PyQt4.QtGui import QSplitter
from model import materials
from utils.gui import table_last_col_fill
from controller.table import tableWithManipulators
from model.grids.section import GridsModel

class GridsController(Controller):

    def __init__(self, document, model = GridsModel()):
        Controller.__init__(self, document, model)
        
        self.splitter = QSplitter()
        
        self.grids_table = QtGui.QTableView()
        self.grids_table.setModel(self.model)
        #self.grids_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model, self.grids_table))
        #self.materialsTableActions = TableActions(self.grids_table)
        table_last_col_fill(self.grids_table, self.model.columnCount(None), 250)
        self.splitter.addWidget(tableWithManipulators(self.grids_table, self.splitter, title="Meshes and generators"))
        
        #self.splitter.addWidget(tableWithManipulators(self.properties_table, self.splitter, title="Properties of the material"))
        
        self.grids_table.selectionModel().selectionChanged.connect(self.grid_selected) #currentChanged ??
        
    def grid_selected(self, newSelection, oldSelection):
        indexes = newSelection.indexes()
        #if indexes:
        #    self.property_model.material = self.model.entries[indexes[0].row()]
        #else:
        #    self.property_model.material = None
        
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