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

from ..qt.QtCore import *
from ..qt import QtSignal
from ..utils.xml import require_no_children, require_no_attributes
from . import SectionModel
from .info import Info


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


class SourceModel(SectionModel):

    def __init__(self, name='source', info_cb=None):
        super().__init__(name, info_cb, undo_stack=_UndoStack(self))
        self._source = ''
        self.editor = None

    def load_xml_element(self, element, undoable=True):
        #TODO undo support (??)
        require_no_children(element)
        require_no_attributes(element)
        if element is not None:
            self.set_text(element.text[1:] if element.text[0] == '\n' else element.text)
        else:
            self.set_text('')

    def get_text(self):
        return self._source

    def set_text(self, text):
        text = text.expandtabs()
        if self._source != text:
            self._source = text
            self.fire_changed()


class ScriptModel(SourceModel):

    def __init__(self, info_cb=None):
        super().__init__('script', info_cb)

    # XML element that represents whole section
    def make_xml_element(self):
        res = etree.Element(self.name)
        if self._source and self._source != '\n':
            code = '\n' + self._source
            if code[-1] != '\n':
                code += '\n'
            res.text = etree.CDATA(code)
        return res
