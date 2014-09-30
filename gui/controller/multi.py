# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from ..qt import QtGui

from ..utils.widgets import exception_to_msg
from .source import SourceEditController
from . import Controller

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

    def change_editor(self):
        if not self.set_current_index(int(self.document.window.showsource_action.isChecked())):
            self.document.window.showsource_action.setChecked(bool(self.get_current_index()))

    def on_edit_enter(self):
        self.document.window.showsource_action.triggered.connect(self.change_editor)
        self.document.window.showsource_action.setEnabled(True)
        super(GUIAndSourceController, self).on_edit_enter()

    def on_edit_exit(self):
        super(GUIAndSourceController, self).on_edit_exit()
        self.document.window.showsource_action.triggered.disconnect(self.change_editor)
        self.document.window.showsource_action.setEnabled(False)
