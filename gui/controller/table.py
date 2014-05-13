from ..qt import QtGui

from .base import Controller

class TableActions(object):

    def __init__(self, table, model=None):
        self.table = table
        self.model = model if model is not None else table.model()

    def add_entry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            row = self.model.insert(index.row()+1)
        else:
            row = self.model.insert()
        if row is not None: self.table.selectRow(row)

    def remove_entry(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            self.model.remove(index.row())

    def move_up(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 1 <= index < len(self.model.entries):
            self.model.swap_neighbour_entries(index-1, index)
            #self.table.selectRow(index-1)

    def move_down(self):
        index = self.table.selectionModel().currentIndex()
        if not index.isValid(): return
        index = index.row()
        if 0 <= index < len(self.model.entries)-1:
            self.model.swap_neighbour_entries(index, index+1)
            #self.table.selectRow(index+1)

    def get(self, parent):
        self.add_action = QtGui.QAction(QtGui.QIcon.fromTheme('list-add', QtGui.QIcon(':/list-add.png')),
                                        '&Add', parent)
        self.add_action.setStatusTip('Add new entry to the list')
        self.add_action.triggered.connect(self.add_entry)

        self.remove_action = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove', QtGui.QIcon(':/list-remove.png')),
                                           '&Remove', parent)
        self.remove_action.setStatusTip('Remove selected entry from the list')
        self.remove_action.triggered.connect(self.remove_entry)

        self.move_up_action = QtGui.QAction(QtGui.QIcon.fromTheme('go-up', QtGui.QIcon(':/go-up.png')),
                                            'Move &up', parent)
        self.move_up_action.setStatusTip('Change order of entries: move current entry up')
        self.move_up_action.triggered.connect(self.move_up)

        self.move_down_action = QtGui.QAction(QtGui.QIcon.fromTheme('go-down', QtGui.QIcon(':/go-down.png')),
                                              'Move &down', parent)
        self.move_down_action.setStatusTip('Change order of entries: move current entry down')
        self.move_down_action.triggered.connect(self.move_down)

        return self.add_action, self.remove_action, self.move_up_action, self.move_down_action

def table_with_manipulators(table, parent=None, model=None, title=None):
    toolbar = QtGui.QToolBar()
    table.table_manipulators_actions = TableActions(table, model)
    toolbar.addActions(table.table_manipulators_actions.get(parent))

    vbox = QtGui.QVBoxLayout()
    vbox.addWidget(toolbar)
    vbox.addWidget(table)

    external = QtGui.QGroupBox()
    if title is not None:
        external.setTitle(title)
        m = external.getContentsMargins()
        external.setContentsMargins(0, m[1], 0, m[3])
    else:
        external.setContentsMargins(0, 0, 0, 0)
    vbox.setContentsMargins(0, 0, 0, 0)

    external.setLayout(vbox)
    #if title is None:
    #widget.setContentsMargins(0, 0, 0, 0)

    return external


class TableController(Controller):

    def __init__(self, document, model):
        Controller.__init__(self, document, model)
        self.table = QtGui.QTableView()
        self.table.setModel(self.model)
        self.table_actions = TableActions(self.table, self.model)

        cols = self.model.columnCount(None)  # column widths:
        for c in range(0, cols-1):
            self.table.setColumnWidth(c, 200)
            #self.table.horizontalHeader().setResizeMode(c, QtGui.QHeaderView.ResizeToContents);
        self.table.horizontalHeader().setResizeMode(cols-1, QtGui.QHeaderView.Stretch)

        self.table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)

    def get_editor(self):
        return self.table

    def on_edit_enter(self):
        self.save_data_in_model()  # This should do nothing, but is called in case of subclass using it
        if not self.model.is_read_only():
            self.document.window.set_section_actions(*self.get_table_edit_actions())
        self.model.changed.connect(self._document_changed)

    # When editor is turned off, model should be updated
    def on_edit_exit(self):
        self.model.changed.disconnect(self._document_changed)
        self.document.window.set_section_actions()

    def get_table_edit_actions(self):
        return self.table_actions.get(self.document.window)
