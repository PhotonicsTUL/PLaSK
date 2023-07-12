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

import weakref
from lxml import etree

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from ..utils import sorted_index


class MuteChanges:

    def __init__(self, controller):
        self.controller = weakref.proxy(controller)

    def __enter__(self):
        self.controller._notify_changes = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.controller._notify_changes = True


class Controller:
    """
        Base class for controllers.
        Controllers create editor for the fragment of the XPL document (a section or smaller fragment) and make it
        available by get_widget method (subclasses must implement this method).
        They also transfer data from model to editor (typically in on_edit_enter) and in opposite direction
        (typically in save_data_in_model).
    """

    def __init__(self, document=None, model=None):
        """
        Optionally set document and/or model.
        :param XPLDocument document: document
        :param model.SectionModel model: model
        """
        super().__init__()
        if document is not None:
            if isinstance(document, weakref.ProxyType):
                self.document = document
            else:
                self.document = weakref.proxy(document)
        if model is not None:
            self.model = model
        self._notify_changes = True

    def save_data_in_model(self):
        """Called to force save data from editor in model (typically by on_edit_exit or when model is needed while
        editor still is active - for instance when user saves edited document to file)."""
        pass

    def can_save(self):
        """Called before document is saved to file."""
        return True

    def get_contents(self):
        section_string = ''
        element = self.model.make_file_xml_element()
        if len(element) or element.text:
            section_string = etree.tostring(element, encoding='unicode', pretty_print=True)
        return section_string

    def on_edit_enter(self):
        """Called when editor is entered and will be visible."""
        pass

    def on_edit_exit(self):
        """Called when editor is left and will be not visible. Typically and by default it calls save_data_in_model."""
        self.save_data_in_model()
        return True

    def update_line_numbers(self, current_line_in_file):
        """If the script has a source editor, update its line numbers offset"""
        self.model.line_in_file = current_line_in_file
        try:
            self.source_widget.editor.line_numbers.offset = current_line_in_file
        except AttributeError:
            pass
        else:
            self.source_widget.editor.repaint()

    def get_widget(self):
        raise NotImplementedError("Method 'get_widget' must be overridden in a subclass!")

    def fire_changed(self, *args, **kwargs):
        if self._notify_changes:
            self.model.fire_changed()

    def select_info(self, info):
        """
        Set focus on widget or cell connecting with the given info object, to help user fixed the problem connecting
        with the this info.
        :param ..model.info.Info info: info object
        """
        pass

    def mute_changes(self):
        return MuteChanges(self)


class NoConfController(Controller):
    """Controller for all models that does not require any configuration."""
    def __init__(self, text='Configuration is neither required nor available.', document=None, model=None):
        super().__init__(document=document, model=model)
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def get_widget(self):
        return self.label


def select_index_from_info(info, model, table):
    try:
        col = info.cols[0]
    except (AttributeError, IndexError):
        col = 0
    try:
        current_row = table.currentIndex().row()
        rows = sorted(info.rows)
        try:
            table.setCurrentIndex(model.createIndex(rows[(sorted_index(rows, current_row)+1) % len(rows)], col))
        except ValueError:
            table.setCurrentIndex(model.createIndex(rows[0], col))
        return True
    except (AttributeError, IndexError):
        return False
