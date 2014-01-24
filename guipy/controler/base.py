from PyQt4 import QtGui
from qhighlighter.XML import XMLHighlighter
from utils import defaultFont, exceptionToMsg

class SourceEditControler(object):

    def __init__(self, document, model):
        object.__init__(self)
        self.document = document
        self.model = model

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

    def onEditEnter(self):
        self.getSourceEditor().setPlainText(self.model.getText())

    # when editor is turn off, model should be update
    def onEditExit(self):
        if not self.getSourceEditor().isReadOnly():
            self.model.setText(self.getSourceEditor().toPlainText())
    
    
            
class MultiEditorControler(object):
    
    def __init__(self, *controlers):
        object.__init__(self)
        self.controlers = list(controlers)
        
        self.editorWidget = QtGui.QStackedWidget()
        for c in controlers:
            self.editorWidget.addWidget(c.getEditor())

    @property
    def model(self):
        return self.controlers[0].model

    @property
    def document(self):
        return self.controlers[0].document
    
    def getEditor(self):
        return self.editorWidget
    
    def getCurrentIndex(self):
        return self.editorWidget.currentIndex()
    
    def setCurrentIndex(self, new_index):
        if self.getCurrentIndex() == new_index: return False;
        if not exceptionToMsg(lambda: self.getCurrectControler().onEditExit(),
                              self.document.mainWindow, 'Error while trying to store data from editor'):
            return False
        self.editorWidget.setCurrentIndex(new_index)
        self.getCurrectControler().onEditEnter()  
        return True     
    
    def getCurrectControler(self):
        return self.controlers[self.getCurrectEditorIndex()]
    
    def onEditEnter(self):
        self.getCurrectControler().onEditEnter()

    def onEditExit(self):
        self.getCurrectControler().onEditExit()
    
    
class GUIAndSourceControler(MultiEditorControler):
    
    def __init__(self, controler):
        MultiEditorControler.__init__(self, controler, SourceEditControler(controler.document, controler.model))
    
    def changeEditor(self):
        if not self.setCurrentIndex(int(self.showSourceAction.isChecked())):
            self.showSourceAction.setChecked(bool(self.getCurrentIndex()))
    
    def getShowSourceAction(self):
        if not hasattr(self, 'showSourceAction'):
            self.showSourceAction = QtGui.QAction(QtGui.QIcon.fromTheme('accessories-text-editor'), '&Show source')
            self.showSourceAction.setCheckable(True)
            self.showSourceAction.setStatusTip('Show XPL source of the current section')
            self.showSourceAction.triggered.connect(self.changeEditor)
        return self.showSourceAction
    
    def onEditEnter(self):
        self.document.mainWindow.setEditorSelectActions(self.getShowSourceAction())
        MultiEditorControler.onEditEnter()

    def onEditExit(self):
        MultiEditorControler.onEditExit()
        self.document.mainWindow.setEditorSelectActions()