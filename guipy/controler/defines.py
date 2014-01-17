from PyQt4 import QtGui
from controler.base import SourceEditControler
from model.defines import DefinesModel
from utils import showError

class DefinesControler(SourceEditControler):

    def __init__(self, main, model = DefinesModel()):
        SourceEditControler.__init__(self, model)
        self.editorWidget = QtGui.QStackedWidget()
        self.table = QtGui.QTableView()
        self.table.setModel(self.model)
        self.editorWidget.addWidget(self.table)
        self.editorWidget.addWidget(self.getSourceEditor())
        self.main = main
        
    def isInSourceEditor(self):
        return self.editorWidget.currentIndex() == 1
    
    def initCurrentEditor(self):
        if self.isInSourceEditor():
            self.getSourceEditor().setPlainText(self.model.getText())
            self.main.setSectionActions(self.getShowSourceAction(self.main))
        else:
            self.main.setSectionActions(self.getShowSourceAction(self.main), None, *self.getTableEditActions(self.main))
        
    def changeEditor(self):
        new_index = int(self.showSourceAction.isChecked())
        if new_index == self.editorWidget.currentIndex(): return
        if self.isInSourceEditor():  # source editor exit
            try:
                self.model.setText(self.getSourceEditor().toPlainText())
            except Exception as e:
                self.showSourceAction.setChecked(True)
                showError(str(e), self.getSourceEditor(), 'Error in source')
                return
        self.editorWidget.setCurrentIndex(new_index)
        self.initCurrentEditor()
        
    def getShowSourceAction(self, parent):
        if not hasattr(self, 'showSourceAction'):
            self.showSourceAction = QtGui.QAction(QtGui.QIcon.fromTheme('accessories-text-editor'), '&Show source', parent)
            self.showSourceAction.setCheckable(True)
            self.showSourceAction.setStatusTip('Show XPL source of the current section')
            self.showSourceAction.triggered.connect(self.changeEditor)
        return self.showSourceAction

    def getEditor(self):
        return self.editorWidget

    def onEditEnter(self, main):
        self.initCurrentEditor()

    # when editor is turn off, model should be update
    def onEditExit(self, main):
        if self.isInSourceEditor(): self.model.setText(self.getSourceEditor().toPlainText())
        main.setSectionActions()
        
    def addDefine(self):
        index = self.table.selectionModel().currentIndex()
        if index.isValid():
            row = self.model.insert(index.row()+1)
        else:
            row = self.model.insert()
        self.table.selectRow(row)
    
    def removeDefine(self):
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
    
    def getTableEditActions(self, parent):
            self.addAction = QtGui.QAction(QtGui.QIcon.fromTheme('list-add'), '&Add definition', parent)
            self.addAction.setStatusTip('Add new definition to the list')
            self.addAction.triggered.connect(self.addDefine)
            
            self.removeAction = QtGui.QAction(QtGui.QIcon.fromTheme('list-remove'), '&Remove definition', parent)
            self.removeAction.setStatusTip('Remove selected definition from the list')
            self.removeAction.triggered.connect(self.removeDefine)
            
            self.moveUpAction = QtGui.QAction(QtGui.QIcon.fromTheme('go-up'), 'Move &up', parent)
            self.moveUpAction.setStatusTip('Change order of entries: move current entry up')
            self.moveUpAction.triggered.connect(self.moveUp)
            
            self.moveDownAction = QtGui.QAction(QtGui.QIcon.fromTheme('go-down'), 'Move &down', parent)
            self.moveDownAction.setStatusTip('Change order of entries: move current entry down')
            self.moveDownAction.triggered.connect(self.moveDown)
            
            return self.addAction, self.removeAction, self.moveUpAction, self.moveDownAction