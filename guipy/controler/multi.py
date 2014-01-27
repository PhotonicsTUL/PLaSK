from PyQt4 import QtGui
from utils import exceptionToMsg
from controler.source import SourceEditControler

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
        return self.controlers[self.getCurrentIndex()]
    
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
            self.showSourceAction = QtGui.QAction(QtGui.QIcon.fromTheme('accessories-text-editor'), '&Show source', self.document.mainWindow)
            self.showSourceAction.setCheckable(True)
            self.showSourceAction.setStatusTip('Show XPL source of the current section')
            self.showSourceAction.triggered.connect(self.changeEditor)
        return self.showSourceAction
    
    def onEditEnter(self):
        self.document.mainWindow.setEditorSelectActions(self.getShowSourceAction())
        super(GUIAndSourceControler, self).onEditEnter()

    def onEditExit(self):
        super(GUIAndSourceControler, self).onEditExit()
        self.document.mainWindow.setEditorSelectActions()