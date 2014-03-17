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
            self.editorWidget.addWidget(c.get_editor())

    @property
    def model(self):
        return self.controllers[0].model

    @property
    def document(self):
        return self.controllers[0].document
    
    def get_editor(self):
        return self.editorWidget
    
    def getCurrentIndex(self):
        return self.editorWidget.currentIndex()
    
    def setCurrentIndex(self, new_index):
        if self.getCurrentIndex() == new_index: return False;
        if not exceptionToMsg(lambda: self.getCurrectController().on_edit_exit(),
                              self.document.mainWindow, 'Error while trying to store data from editor'):
            return False
        self.editorWidget.setCurrentIndex(new_index)
        self.getCurrectController().on_edit_enter()  
        return True     
    
    def getCurrectController(self):
        return self.controllers[self.getCurrentIndex()]
    
    def save_data_in_model(self):
        self.getCurrectController().save_data_in_model()
    
    def on_edit_enter(self):
        self.getCurrectController().on_edit_enter()

    def on_edit_exit(self):
        self.getCurrectController().on_edit_exit()
    
    
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
    
    def on_edit_enter(self):
        self.document.mainWindow.setEditorSelectActions(self.getShowSourceAction())
        super(GUIAndSourceController, self).on_edit_enter()

    def on_edit_exit(self):
        super(GUIAndSourceController, self).on_edit_exit()
        self.document.mainWindow.setEditorSelectActions()