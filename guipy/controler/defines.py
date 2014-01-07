from PyQt4 import QtGui
from controler.base import SourceEditControler
from model.defines import DefinesModel

class DefinesControler(SourceEditControler):

    def __init__(self, model = DefinesModel()):
        SourceEditControler.__init__(self, model)
        self.editorWidget = QtGui.QStackedWidget()
        self.table = QtGui.QTableView()
        self.table.setModel(self.model)
        self.editorWidget.addWidget(self.table)
        self.editorWidget.addWidget(self.getSourceEditor())
        
    def changeEditor(self):
        self.editorWidget.setCurrentIndex(int(self.showSourceAction.isChecked()))
        
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
        main.setSectionActions(self.getShowSourceAction(main))
        self.getSourceEditor().setPlainText(self.model.getText())

    # when editor is turn off, model should be update
    def onEditExit(self, main):
        self.model.setText(self.getSourceEditor().toPlainText())
        main.setSectionActions()