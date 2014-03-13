from PyQt4 import QtGui
from qhighlighter.XML import XMLHighlighter
from utils.gui import defaultFont
from controller.base import Controller

class SourceEditController(Controller):

    def __init__(self, document, model):
        Controller.__init__(self, document, model)

    def createSourceEditor(self, parent = None):
        ed = QtGui.QTextEdit(parent)
        ed.setFont(defaultFont)
        self.highlighter = XMLHighlighter(ed.document())   # highlighter varible is required, in other case it is deleted and text is not highlighted
        ed.setReadOnly(self.model.isReadOnly())
        return ed

    # text, source editor
    def getSourceEditor(self, parent = None):
        if not hasattr(self, 'sourceEditor'): self.sourceEditor = self.createSourceEditor(parent)
        return self.sourceEditor

    # GUI editor, by default use source editor
    def getEditor(self):
        return self.getSourceEditor()

    def refreshEditor(self, *ignore):
        self.getSourceEditor().setPlainText(self.model.getText())
        
    def saveDataInModel(self):
        if not self.getSourceEditor().isReadOnly():
            self.model.setText(self.getSourceEditor().toPlainText())

    def onEditEnter(self):
        self.refreshEditor()
        self.model.changed += self.refreshEditor 

    # when editor is turn off, model should be update
    def onEditExit(self):
        self.saveDataInModel()
        self.model.changed -= self.refreshEditor