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

from ...qt import QtCore, QtGui
from ...qt.QtCore import Qt

from ...model.script.completer import CompletionsModel, get_completions


class CompletionsPopup(QtGui.QTableView):

    def __init__(self, textedit):
        super(CompletionsPopup, self).__init__(None)
        self.setMinimumHeight(150)
        self._textedit = textedit
        # self.setAlternatingRowColors(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setSortingEnabled(False)
        self.setShowGrid(False)
        self.setWordWrap(False)

    visibility_changed = QtCore.Signal(bool)

    def showEvent(self, event):
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        self.visibility_changed.emit(False)


class CompletionsController(QtGui.QCompleter):

    def __init__(self, textedit, sorting=False, case_sensitivity=False):
        super(CompletionsController, self).__init__(textedit)
        self.textedit = textedit
        self.popup = CompletionsPopup(self.textedit)
        self.popup.visibility_changed[bool].connect(self.popup_visibility_changed)
        self.setPopup(self.popup)
        self.setCompletionColumn(1)
        # self.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setWidget(self.textedit)
        self.setCaseSensitivity(Qt.CaseSensitive if case_sensitivity else Qt.CaseInsensitive)
        self.activated[str].connect(self.insert_completion)
        self.textedit.removeEventFilter(self)

    def text_under_cursor(self):
        tc = self.textedit.textCursor()
        tc.select(QtGui.QTextCursor.WordUnderCursor)
        return tc.selectedText()

    def insert_completion(self, completion):
        tc = self.textedit.textCursor()
        prefix_length = len(unicode(self.text_under_cursor()))
        suffix = unicode(completion)[prefix_length:]
        tc.insertText(suffix)
        self.textedit.setTextCursor(tc)

    def popup_visibility_changed(self, is_visible):
        if is_visible:
            self.textedit.installEventFilter(self)
        else:
            self.textedit.removeEventFilter(self)

    def eventFilter(self, obj, event):
        if isinstance(event, QtGui.QKeyEvent) and self.key_pressed(event):
            return True
        return super(CompletionsController, self).eventFilter(obj, event)

    def key_pressed(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.completionMode() != QtGui.QCompleter.UnfilteredPopupCompletion:
                sel_list = self.popup.selectionModel().selectedIndexes()
                if sel_list:
                    idx = sel_list[0]
                else:
                    idx = None
            else:
                idx = self.popup.currentIndex()
            if idx is not None:
                activated = self.completionModel().data(idx, Qt.DisplayRole)
                self.insert_completion(activated)
            self.popup.hide()
            return True
        if event.text():
            prefix = self.text_under_cursor()
            self.setCompletionPrefix(prefix)
            self.popup.setCurrentIndex(self.completionModel().index(0, 0))
            return
        elif event.key() in (Qt.Key_Shift, Qt.Key_Control,
                             Qt.Key_Backspace,
                             Qt.Key_Down, Qt.Key_Up, Qt.Key_PageDown, Qt.Key_PageUp):
            return
        else:
            self.popup.hide()
            return True

    def show_completions(self):
        cursor = self.textedit.textCursor()
        items = get_completions(self.textedit.toPlainText(), cursor.blockNumber(), cursor.positionInBlock())

        if not items:
            return
        elif len(items) == 1:
            self.insert_completion(items[0].name)
            return

        self.model = CompletionsModel(items)
        self.setModel(self.model)
        self.popup.resizeColumnsToContents()
        self.popup.resizeRowsToContents()
        self.popup.updateGeometries()
        #self.popup.sortByColumn(1, Qt.AscendingOrder)
        #self.popup.setModel(self._model)
        #self.popup.setCurrentIndex(self._model.index(0, 0))
        self.setCurrentRow(0)

        cr = self.textedit.cursorRect()
        cr.setWidth(self.popup.sizeHintForColumn(0))

        self.popup.setCurrentIndex(self.completionModel().index(0, 0))
        self.complete(cr)
