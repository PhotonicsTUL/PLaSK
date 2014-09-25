from ..qt import QtGui

from ..utils.gui import exception_to_msg
from .source import SourceEditController
from .base import Controller

class MultiEditorController(Controller):
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

    def get_current_index(self):
        """:return: an index of current controller (int)"""
        return self.editorWidget.currentIndex()

    def set_current_index(self, new_index):
        """
            Try to change current controller.
            :param int new_index: index of new current controller
            :return: true only when controller has been changed (bool)
        """
        if self.get_current_index() == new_index: return False
        if not exception_to_msg(lambda: self.currect_controller.on_edit_exit(),
                              self.document.window, 'Error while trying to store data from editor'):
            return False
        self.editorWidget.setCurrentIndex(new_index)
        self.currect_controller.on_edit_enter()
        return True

    @property
    def currect_controller(self):
        """:return: current controller"""
        return self.controllers[self.get_current_index()]

    def save_data_in_model(self):
        self.currect_controller.save_data_in_model()

    def on_edit_enter(self):
        self.currect_controller.on_edit_enter()

    def on_edit_exit(self):
        self.currect_controller.on_edit_exit()


class GUIAndSourceController(MultiEditorController):

    def __init__(self, controller):
        MultiEditorController.__init__(self, controller, SourceEditController(controller.document, controller.model))

    def changeEditor(self):
        if not self.set_current_index(int(self.showSourceAction.isChecked())):
            self.showSourceAction.setChecked(bool(self.get_current_index()))

    def getShowSourceAction(self):
        if not hasattr(self, 'showSourceAction'):
            self.showSourceAction = QtGui.QAction(
                QtGui.QIcon.fromTheme('accessories-text-editor', QtGui.QIcon(':/accessories-text-editor.png')),
                '&Show source', self.document.window)
            self.showSourceAction.setCheckable(True)
            self.showSourceAction.setStatusTip('Show XPL source of the current section')
            self.showSourceAction.triggered.connect(self.changeEditor)
        return self.showSourceAction

    def on_edit_enter(self):
        self.document.window.set_editor_select_actions(self.getShowSourceAction())
        super(GUIAndSourceController, self).on_edit_enter()

    def on_edit_exit(self):
        super(GUIAndSourceController, self).on_edit_exit()
        self.document.window.set_editor_select_actions()
