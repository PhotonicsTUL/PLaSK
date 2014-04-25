from ..qt import QtCore, QtGui
from ..qt.QtGui import QSplitter

from ..model.materials import MaterialsModel, MaterialPropertyModel, materialHTMLHelp
from ..utils.gui import HTMLDelegate, table_last_col_fill
from .base import Controller
from .defines import DefinesCompletionDelegate
from .table import table_with_manipulators

try:
    import plask
except ImportError:
    pass

class MaterialBaseDelegate(DefinesCompletionDelegate):

    def __init__(self, definesModel, parent):
        DefinesCompletionDelegate.__init__(self, definesModel, parent)

    def createEditor(self, parent, option, index):

        earlier_names = ['semiconductor', 'metal', 'dielectric', 'liquid_crystal']

        try:
            earlier_names.extend(sorted(mat for mat in plask.material.db if mat not in earlier_names))
        except NameError:
            pass

        earlier_names.extend(e.name for e in index.model().entries[0:index.row()])

        if not earlier_names: return super(MaterialBaseDelegate, self).createEditor(parent, option, index)

        combo = QtGui.QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(earlier_names)
        combo.insertSeparator(5)
        combo.insertSeparator(len(earlier_names)-index.row()+1)
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
        combo.setAutoCompletionCaseSensitivity(True)
        combo.highlighted.connect(lambda i:
            QtGui.QToolTip.showText(QtGui.QCursor.pos(), materialHTMLHelp(combo.itemText(i)))
        )
        #combo.setCompleter(completer)
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
        #             self, QtCore.SLOT("currentIndexChanged()"))
        return combo


class MaterialsController(Controller):

    def __init__(self, document, selection_model=MaterialsModel()):
        Controller.__init__(self, document, selection_model)

        self.splitter = QSplitter()

        self.materials_table = QtGui.QTableView()
        self.materials_table.setModel(self.model)
        self.materials_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model, self.materials_table))
        #self.materialsTableActions = TableActions(self.materials_table)
        table_last_col_fill(self.materials_table, self.model.columnCount(None), 140)
        self.splitter.addWidget(table_with_manipulators(self.materials_table, self.splitter, title="Materials"))

        self.property_model = MaterialPropertyModel(selection_model)
        self.properties_table = QtGui.QTableView()
        self.properties_table.setModel(self.property_model)
        self.properties_delegate = MaterialPropertiesDelegate(self.document.defines.model, self.properties_table)
        self.unit_delegate = HTMLDelegate()
        self.help_delegate = HTMLDelegate()
        self.properties_table.setItemDelegateForColumn(0, self.properties_delegate)
        self.properties_table.setItemDelegateForColumn(1, self.properties_delegate)
        self.properties_table.setItemDelegateForColumn(2, self.unit_delegate)
        self.properties_table.setItemDelegateForColumn(3, self.help_delegate)
        #self.properties_table.setWordWrap(True)
        table_last_col_fill(self.properties_table, self.property_model.columnCount(None), [90, 180, 50])

        self.properties_table.verticalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.splitter.addWidget(table_with_manipulators(self.properties_table, self.splitter, title="Properties of the material"))

        self.splitter.setSizes([10000,30000])

        self.materials_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.materials_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        selection_model = self.materials_table.selectionModel()
        selection_model.selectionChanged.connect(self.material_selected) #currentChanged ??

        self.properties_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.properties_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

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
