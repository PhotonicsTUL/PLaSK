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

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from ..utils import sorted_index


class MuteChanges(object):

    def __init__(self, controller):
        self.controller = controller

    def __enter__(self):
        self.controller._notify_changes = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.controller._notify_changes = True


class Controller(object):
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
        super(Controller, self).__init__()
        if document is not None:
            self.document = document
        if model is not None:
            self.model = model
        self._notify_changes = True

    def save_data_in_model(self):
        """Called to force save data from editor in model (typically by on_edit_exit or when model is needed while
        editor still is active - for instance when user saves edited document to file)."""
        pass  

    def on_edit_enter(self):
        """Called when editor is entered and will be visible."""
        pass

    def on_edit_exit(self):
        """Called when editor is left and will be not visible. Typically and by default it calls save_data_in_model."""
        return self.try_save_data_in_model()

    def update_line_numbers(self, current_line_in_file):
        """If the script has a source editor, update its line numbers offset"""
        self.model.line_in_file = current_line_in_file
        try:
            self.source_widget.editor.line_numbers.offset = current_line_in_file
        except AttributeError:
            pass
        else:
            self.source_widget.editor.repaint()

    def try_save_data_in_model(self, cat=None):
        try:
            self.save_data_in_model()
        except Exception as exc:
            from .. import _DEBUG
            if _DEBUG:
                import traceback as tb
                tb.print_stack()
                tb.print_exc()
                # sys.stderr.write('Traceback (most recent call last):\n')
                # sys.stderr.write(''.join(tb.format_list(tb.extract_stack()[:-1])))
                # sys.stderr.write('{}: {}\n'.format(exc.__class__.__name__, exc))
            return QMessageBox().critical(
                None, "Error saving data from editor",
                "An error occured while trying to save data from editor:\n"
                "(caused either by wrong values entered or a by a program error)\n\n"
                "{}: {}\n\n"
                "Do you want to discard your changes in this {}editor and move on?"
                .format(exc.__class__.__name__, str(exc), '' if cat is None else (cat+' ')),
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes
        else:
            return True

    def get_widget(self):
        raise NotImplementedError("Method 'get_widget' must be overriden in a subclass!")

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
    def __init__(self, text='Configuration is neither required nor available.'):
        super(NoConfController, self).__init__()
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignCenter)

    def get_widget(self):
        return self.label


# unused now
class ControllerWithSubController(Controller):
    """
        Subclass must have grid field or property (qt table or tree) and implement get_controller_for_index.
        After changing index, on_current_index_changed is called.
    """

    def __init__(self, document, model):
        """self.grid must be available before call this __init__"""
        super(ControllerWithSubController, self).__init__(document, model)

        self._last_index = None
        self._current_index = None
        self._current_controller = None

        self.grid.setModel(self.model)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.parent_for_editor_widget = QStackedWidget()

        selection_model = self.grid.selectionModel()
        selection_model.selectionChanged.connect(self.grid_selected) #currentChanged ??

    def get_controller_for_index(self, index):
        #self.model.entries[new_index].get_controller(self.document)
        return None

    def on_current_index_changed(self, new_index):
        pass

    def set_current_index(self, new_index):
        """
            Try to change current script.
            :param int new_index: index of new current script
            :return: False only when script should restore old selection
        """
        if self._current_index == new_index: return True
        if self._current_controller is not None:
            if not self._current_controller.on_edit_exit():
                return False
        self._current_index = new_index
        for i in reversed(range(self.parent_for_editor_widget.count())):
            self.parent_for_editor_widget.removeWidget(self.parent_for_editor_widget.widget(i))
        if self._current_index is None:
            self._current_controller = None
        else:
            self._current_controller = self.get_controller_for_index(new_index)
            if self._current_controller is not None:
                self.parent_for_editor_widget.addWidget(self._current_controller.get_widget())
                self._current_controller.on_edit_enter()
        self.on_current_index_changed(new_index)
        return True

    def grid_selected(self, new_selection, old_selection):
        if new_selection.indexes() == old_selection.indexes(): return
        indexes = new_selection.indexes()
        if not self.set_current_index(new_index=(indexes[0].row() if indexes else None)):
            self.grid.selectionModel().select(old_selection, QItemSelectionModel.ClearAndSelect)

    def on_edit_enter(self):
        self.grid.selectionModel().clear()   # model could completly changed
        if self._last_index is not None:
            self.grid.selectRow(self._last_index)

    def on_edit_exit(self):
        if self._current_controller is not None:
            self._last_index = self._current_index
            self.grid.selectionModel().clear()
        return True


def select_index_from_info(info, model, table):
    try:
        col = info.cols[0]
    except (AttributeError, IndexError):
        col = 0
    try:
        current_row = table.currentIndex().row()
        rows = sorted(info.rows)
        try:
            table.setCurrentIndex(model.createIndex(rows[(sorted_index(rows, current_row)+1)%len(rows)], col))
        except ValueError:
            table.setCurrentIndex(model.createIndex(rows[0], col))
        return True
    except (AttributeError, IndexError):
        return False
