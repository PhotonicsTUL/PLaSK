# Copyright (C) 2021 Photonics Group, Lodz University of Technology
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

import weakref

from ...qt.QtCore import *
from ...qt.QtGui import *
from ...qt.QtWidgets import *
from ...qt import qt_exec

from ...lib.highlighter.plask import SYNTAX, get_syntax

from ...utils.texteditor.python import PythonTextEditor
from ...utils.qsignals import BlockQtSignals
from .object import GNObjectController


SYNTAX['formats']['geometry_object'] = '{syntax_solver}'


class GNPythonController(GNObjectController):

    have_mesh_settings = False

    def construct_form(self):
        self.construct_group('Python Code')
        weakself = weakref.proxy(self)
        self.editor = self.construct_text_edit(node_property_name='code', display_property_name="python code",
                                               editor_class=PythonTextEditor)
        self.editor.setToolTip('Type Python code here. You should assign the geometry object to insert here '
                               'to the variable <tt>__object__</tt>.')
        self.get_current_form().addRow(self.editor)
        super().construct_form()

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.editor):
            self.editor.setPlainText(self.node.code)
        self.editor.rehighlight(self.document.defines, geometry_object=['__object__'])
