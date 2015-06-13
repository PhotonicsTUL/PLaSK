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

from ...qt import QtGui
from ...qt.QtCore import Qt

from ...utils.qthread import BackgroundTask

from ...model.script.completer import CompletionsModel, get_completions


class CompletionsController(QtGui.QCompleter):

    def __init__(self, edit, parent=None):
        super(CompletionsController, self).__init__(parent)
        self._edit = edit
        self.setWidget(edit)
        self.setCompletionMode(QtGui.QCompleter.PopupCompletion)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.activated.connect(self.insert_completion)
        self.popup().setMinimumWidth(300)
        self.popup().setMinimumHeight(200)
        self.popup().setAlternatingRowColors(True)

    def insert_completion(self, completion):
        # if self.widget() != self._edit: return
        cursor = self._edit.textCursor()
        extra = len(self.completionPrefix())
        if not (cursor.atBlockStart() or
                self._edit.document().characterAt(cursor.position()-1).isspace()):
            cursor.movePosition(QtGui.QTextCursor.Left)
        cursor.movePosition(QtGui.QTextCursor.EndOfWord)
        cursor.insertText(completion[extra:])
        self._edit.setTextCursor(cursor)

    def start_completion(self):
        cursor = self._edit.textCursor()
        row = cursor.blockNumber()
        col = cursor.positionInBlock()
        cursor.select(QtGui.QTextCursor.WordUnderCursor)
        completion_prefix = cursor.selectedText()

        def thread_finished(completions):
            tc = self._edit.textCursor()
            if tc.blockNumber() == row and tc.positionInBlock() == col:
                self.show_completion_popup(completion_prefix, completions)
            QtGui.QApplication.restoreOverrideCursor()

        QtGui.QApplication.setOverrideCursor(Qt.BusyCursor)
        task = BackgroundTask(
            lambda: get_completions(self._edit.controller.document, self._edit.toPlainText(), row, col),
            thread_finished)
        task.start()

    def show_completion_popup(self, completion_prefix, completions):
        if completions:
            self.setModel(CompletionsModel(completions))
            if completion_prefix != self.completionPrefix():
                self.setCompletionPrefix(completion_prefix)
                self.popup().setCurrentIndex(self.completionModel().index(0, 0))
            rect = self._edit.cursorRect()
            rect.setWidth(self.popup().sizeHintForColumn(0) + self.popup().verticalScrollBar().sizeHint().width())
            self.complete(rect)  # popup it up!
