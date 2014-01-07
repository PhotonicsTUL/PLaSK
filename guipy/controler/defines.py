from PyQt4 import QtGui
from qhighlighter.XML import XMLHighlighter
from controler.base import SourceEditControler
from model.defines import DefinesModel

class DefinesControler(SourceEditControler):

    def __init__(self, model = DefinesModel()):
        object.__init__(self)
        self.model = model
        
    #TODO:

    # GUI editor, by default use source editor
    def getEditor(self):
        return self.getSourceEditor()

    def onEditEnter(self):
        self.getSourceEditor().setPlainText(self.model.getText())

    # when editor is turn off, model should be update
    def onEditExit(self):
        self.model.setText(self.getSourceEditor().toPlainText())