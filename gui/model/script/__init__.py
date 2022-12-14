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

from ...qt.QtCore import *
from ...qt import QtSignal
from ...utils.xml import require_no_children, require_no_attributes
from .. import SectionModel
from ..info import Info


class _UndoStack(QObject):

    cleanChanged = QtSignal(bool)

    def __init__(self, model):
        super().__init__()
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
        self.editor = None

    def load_xml_element(self, element, undoable=True):
        #TODO undo support (??)
        require_no_children(element)
        require_no_attributes(element)
        if element is not None:
            self.set_text(element.text[1:] if element.text[0] == '\n' else element.text)
        else:
            self.set_text('')

    # XML element that represents whole section
    def make_xml_element(self):
        res = etree.Element(self.name)
        if self._code and self._code != '\n':
            code = '\n' + self._code
            if code[-1] != '\n':
                code += '\n'
            res.text = etree.CDATA(code)
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
                     align=Qt.AlignmentFlag.AlignRight)]
