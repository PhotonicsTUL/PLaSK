# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from lxml import etree

from ..qt.QtWidgets import *
from ..qt import qt_exec

from ..utils.widgets import fire_edit_end
from ..utils.qsignals import BlockQtSignals
from . import Controller
from .source import SourceEditController
from .geometry.source import GeometrySourceController


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

    def __init__(self, controller, source=None):
        if source is None:
            source = SourceEditController(controller.document, controller.model)
        self.gui = controller
        self.source = source
        super().__init__(controller, source)

    def change_editor(self):
        fire_edit_end()
        if self.document.window.showsource_action.isChecked():
            self.gui.on_edit_exit()
            self.editorWidget.setCurrentIndex(1)
            self.source.on_edit_enter()
        else:
            if self.source.on_edit_exit():
                self.editorWidget.setCurrentIndex(0)
                self.gui.on_edit_enter()
            else:
                with BlockQtSignals(self.document.window.showsource_action):
                    self.document.window.showsource_action.setChecked(True)
                msgbox = QMessageBox()
                msgbox.setText("Edited content of this section is invalid. Cannot switch to GUI mode.")
                msgbox.setDetailedText(str(self.source.error))
                msgbox.setStandardButtons(QMessageBox.StandardButton.Ok)
                msgbox.setIcon(QMessageBox.Icon.Warning)
                msgbox.exec()

    def on_edit_enter(self):
        if self.source.error is not None:
            self.document.window.set_show_source_state(True)
        self.document.window.showsource_action.triggered.connect(self.change_editor)
        self.editorWidget.setCurrentIndex(int(self.document.window.get_show_source_state(do_enabled=True)))
        super().on_edit_enter()

    def on_edit_exit(self):
        result = super().on_edit_exit()
        try: self.document.window.showsource_action.triggered.disconnect(self.change_editor)
        except: pass
        self.document.window.set_show_source_state(None)
        return True

    def get_source_widget(self):
        return self.controllers[1].get_source_widget()

    def get_contents(self):
        if self.source.error is None:
            return super().get_contents()
        else:
            return self.source.error_data

    def can_save(self):
        if not self.source.error:
            return True
        else:
            name = self.document.SECTION_NAMES[self.document.controllers.index(self)]
            msgbox = QMessageBox()
            msgbox.setText("Edited content of {} section is invalid.".format(name))
            msgbox.setDetailedText(str(self.source.error))
            msgbox.setInformativeText("Do you want to save anyway?")
            msgbox.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msgbox.setIcon(QMessageBox.Icon.Warning)
            msgbox.setDefaultButton(QMessageBox.StandardButton.Yes)
            return qt_exec(msgbox) == QMessageBox.StandardButton.Yes

class GeometryGUIAndSourceController(GUIAndSourceController):

    def __init__(self, controller):
        source = GeometrySourceController(controller.document, controller.model)
        super().__init__(controller, source)
