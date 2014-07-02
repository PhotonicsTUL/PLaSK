from ...qt import QtGui
from ...qt.QtGui import QSplitter, QItemSelectionModel

from ..base import Controller
from ...utils.gui import table_last_col_fill, exception_to_msg
from ..table import table_with_manipulators
from ...model.grids.section import GridsModel


class GridsController(Controller):

    def __init__(self, document, model=None):
        if model is None: model = GridsModel()
        Controller.__init__(self, document, model)

        self.current_index = None
        self.current_controller = None

        self.splitter = QSplitter()

        self.grids_table = QtGui.QTableView()
        self.grids_table.setModel(self.model)
        #self.grids_table.setItemDelegateForColumn(1, MaterialBaseDelegate(self.document.defines.model, self.grids_table))
        #self.materialsTableActions = TableActions(self.grids_table)
        table_last_col_fill(self.grids_table, self.model.columnCount(None), 80)
        self.grids_table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.grids_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.splitter.addWidget(table_with_manipulators(self.grids_table, self.splitter, title="Meshes and generators"))

        self.parent_for_editor_widget = QtGui.QStackedWidget()
        self.splitter.addWidget(self.parent_for_editor_widget)

        self.splitter.setSizes([10000,26000])

        selection_model = self.grids_table.selectionModel()
        selection_model.selectionChanged.connect(self.grid_selected) #currentChanged ??

    def set_current_index(self, new_index):
        """
            Try to change current controller.
            :param int new_index: index of new current controller
            :return: False only when controller should restore old selection
        """
        if self.current_index == new_index: return True
        if self.current_controller is not None:
            if not exception_to_msg(lambda: self.current_controller.on_edit_exit(),
                              self.document.window, 'Error while trying to store data from current grid editor'):
                return False
        self.current_index = new_index
        for i in reversed(range(self.parent_for_editor_widget.count())):
            self.parent_for_editor_widget.removeWidget(self.parent_for_editor_widget.widget(i))
        if self.current_index is None:
            self.current_controller = None
        else:
            self.current_controller = self.model.entries[new_index].get_controller(self.document)
            self.parent_for_editor_widget.addWidget(self.current_controller.get_editor())
            self.current_controller.on_edit_enter()
        return True

    def grid_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0].row() if indexes else None)):
            self.grids_table.selectionModel().select(old_selection, QItemSelectionModel.ClearAndSelect)

    def get_editor(self):
        return self.splitter

    #def onEditEnter(self):
    #    self.saveDataInModel()  #this should do nothing, but is called in case of subclass use it
    #    if not self.model.isReadOnly():
    #        self.document.window.setSectionActions(*self.get_table_edit_actions())

    # when editor is turn off, model should be update
    #def onEditExit(self):
    #    self.document.window.setSectionActions()

    def get_table_edit_actions(self):
        return self.tableActions.get(self.document.window)
