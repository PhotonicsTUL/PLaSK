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

from lxml import etree

from ...qt.QtCore import Qt, QObject
from ...qt import QtSignal
from ...utils.xml import require_no_children, require_no_attributes
from .. import SectionModel
from ..info import Info
from .completer import prepare_completions


class _UndoStack(QObject):

    cleanChanged = QtSignal(bool)

    def __init__(self, model):
        super(_UndoStack, self).__init__()
        self.model = model

    def isClean(self):
        return self.model.editor is None or not self.model.editor.document().isModified()

    def setClean(self):
        if self.model.editor is not None:
            self.model.editor.document().setModified(False)


class ScriptModel(SectionModel):

    def __init__(self, info_cb=None):
        SectionModel.__init__(self, 'script', info_cb, undo_stack=_UndoStack(self))
        self._code = ''
        prepare_completions()
        self.editor = None

    def set_xml_element(self, element, undoable=True):
        #TODO undo support (??)
        require_no_children(element)
        require_no_attributes(element)
        if element is not None:
            self.set_text(element.text[1:] if element.text[0] == '\n' else element.text)
        else:
            self.set_text('')

    # XML element that represents whole section
    def get_xml_element(self):
        res = etree.Element(self.name)
        if self._code and self._code != '\n':
            res.text = etree.CDATA('\n' + self._code)
        return res

    def get_text(self):
        return self._code

    def set_text(self, text):
        self._code = text.expandtabs()
        self.fire_changed()

    def create_info(self):
        if self.editor is None: return []
        line_in_file = 0 if self.line_in_file is None else self.line_in_file
        cursor = self.editor.textCursor()
        return [Info('{}:{}   '.format(cursor.blockNumber()+line_in_file+1, cursor.columnNumber()+1),
                     align=Qt.AlignRight)]
