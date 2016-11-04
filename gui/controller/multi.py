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

from ..qt.QtWidgets import *

from ..utils.widgets import fire_edit_end
from . import Controller
from .source import SourceEditController

class MultiEditorController(Controller):
    """
        Controller which consist with a list of controllers and display one at time (using QStackedWidget).
        Allows to change current script.
    """

    def __init__(self, *controllers):
        object.__init__(self)
        self.controllers = list(controllers)

        self.editorWidget = QStackedWidget()
        for c in controllers:
            self.editorWidget.addWidget(c.get_widget())

    def __getitem__(self, i):
        return self.controllers[i]

    @property
    def model(self):
        return self.controllers[0].model

    @property
    def document(self):
        return self.controllers[0].document

    def get_widget(self):
        return self.editorWidget

    def get_current_index(self):
        """:return: an index of current script (int)"""
        return self.editorWidget.currentIndex()

    def set_current_index(self, new_index):
        """
            Try to change current script.
            :param int new_index: index of new current script
            :return: true only when script has been changed (bool)
        """
        if self.get_current_index() == new_index: return False
        if not self.current_controller.on_edit_exit():
            return False
        self.editorWidget.setCurrentIndex(new_index)
        self.current_controller.on_edit_enter()
        return True

    @property
    def current_controller(self):
        """:return: current script"""
        return self.controllers[self.get_current_index()]

    def save_data_in_model(self):
        self.current_controller.save_data_in_model()

    def on_edit_enter(self):
        self.current_controller.on_edit_enter()

    def on_edit_exit(self):
        return self.current_controller.on_edit_exit()

    def select_info(self, info):
        self.current_controller.select_info(info)


class GUIAndSourceController(MultiEditorController):

    def __init__(self, controller):
        source = SourceEditController(controller.document, controller.model)
        MultiEditorController.__init__(self, controller, source)
        self.gui = controller
        self.source = source

    def change_editor(self):
        fire_edit_end()
        if not self.set_current_index(int(self.document.window.showsource_action.isChecked())):
            self.document.window.set_show_source_state(bool(self.get_current_index()))

    def on_edit_enter(self):
        self.document.window.showsource_action.triggered.connect(self.change_editor)
        self.editorWidget.setCurrentIndex(int(self.document.window.get_show_source_state(do_enabled=True)))
        super(GUIAndSourceController, self).on_edit_enter()

    def on_edit_exit(self):
        result = super(GUIAndSourceController, self).on_edit_exit()
        self.document.window.showsource_action.triggered.disconnect(self.change_editor)
        self.document.window.set_show_source_state(None)
        return result

    def get_source_widget(self):
        return self.controllers[1].get_source_widget()
