from PyQt4 import QtGui
from utils.gui import exceptionToMsg
from controller.source import SourceEditController

class MultiEditorController(object):
    """
        Controller which consist with a list of controllers and display one at time (using QStackedWidget).
        Allows to change current controller.
    """
    
    def __init__(self, *controllers):
        object.__init__(self)
        self.controllers = list(controllers)
        
        self.editorWidget = QtGui.QStackedWidget()
        for c in controllers:
            self.editorWidget.addWidget(c.getEditor())

    @property
    def model(self):
        return self.controllers[0].model

    @property
    def document(self):
        return self.controllers[0].document
    
    def getEditor(self):
        return self.editorWidget
    
    def getCurrentIndex(self):
        return self.editorWidget.currentIndex()
    
    def setCurrentIndex(self, new_index):
        if self.getCurrentIndex() == new_index: return False;
        if not exceptionToMsg(lambda: self.getCurrectController().onEditExit(),
                              self.document.mainWindow, 'Error while trying to store data from editor'):
            return False
        self.editorWidget.setCurrentIndex(new_index)
        self.getCurrectController().onEditEnter()  
        return True     
    
    def getCurrectController(self):
        return self.controllers[self.getCurrentIndex()]
    
    def saveDataInModel(self):
        self.getCurrectController().saveDataInModel()
    
    def onEditEnter(self):
        self.getCurrectController().onEditEnter()

    def onEditExit(self):
        self.getCurrectController().onEditExit()
    
    
class GUIAndSourceController(MultiEditorController):
    
    def __init__(self, controller):
        MultiEditorController.__init__(self, controller, SourceEditController(controller.document, controller.model))
    
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
        super(GUIAndSourceController, self).onEditEnter()

    def onEditExit(self):
        super(GUIAndSourceController, self).onEditExit()
        self.document.mainWindow.setEditorSelectActions()