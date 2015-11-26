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

import sys

from ...qt import QtCore, QtGui
from ...qt.QtCore import Qt

from ...utils.qthread import BackgroundTask, Lock
from ...utils.config import CONFIG
try:
    import jedi
except ImportError:
    jedi = None


JEDI_MUTEX = QtCore.QMutex()

PREAMBLE = """\
from pylab import *
import plask
from plask import *
import plask.geometry, plask.mesh, plask.material, plask.plow, plask.phys, plask.algorithm
from plask import geometry, mesh, material, flow, phys, algorithm
from plask.pylab import *
from plask.hdf5 import *
"""
def preload_jedi_modules():
    with Lock(JEDI_MUTEX) as lck:
        jedi.Script(PREAMBLE, 8, 0, None).completions()


def prepare_completions():
    if jedi and not CONFIG['workarounds/no_jedi']:
        # from ... import _DEBUG
        # if _DEBUG:
        #     def print_debug(obj, txt):
        #         sys.stderr.write(txt + '\n')
        #     jedi.set_debug_function(print_debug)
        if CONFIG['workarounds/blocking_jedi']:
            preload_jedi_modules()
        else:
            task = BackgroundTask(preload_jedi_modules)
            task.start()


class CompletionsModel(QtCore.QAbstractTableModel):

    _icons = {}

    def __init__(self, items):
        super(CompletionsModel, self).__init__()
        self.items = items

        if not self._icons:
            self._load_icons()

    def _load_icons(self):
        code_function = QtGui.QIcon.fromTheme("code-function")
        code_class = QtGui.QIcon.fromTheme("code-class")
        code_variable = QtGui.QIcon.fromTheme("code-variable")
        code_typedef = QtGui.QIcon.fromTheme("code-typedef")
        code_block = QtGui.QIcon.fromTheme("code-block")
        code_context = QtGui.QIcon.fromTheme("code-context")
        no_icon = QtGui.QIcon()
        self._icons.update({None: no_icon,
                            "function": code_function,
                            "class": code_class,
                            "statement": code_variable,
                            "instance": code_variable,
                            "keyword": code_context,
                            "module": code_block,
                            "import": code_typedef,
                            "flow": code_context,
                           })

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        return ('icon', 'completion')[section]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            row = index.row()
            col = index.column()
            value = self.items[row]
            if role in (Qt.DisplayRole, Qt.EditRole):
                return value[0]
            elif role == Qt.DecorationRole:
                return self._icons.get(value[1], QtGui.QIcon())
            else:
                return None
        else:
            return None

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.items)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 1


def _try_type(compl):
    try:
        return compl.type
    except:
        return None


def get_completions(document, text, block, column):
    if jedi is None or CONFIG['workarounds/no_jedi']: return
    from ... import _DEBUG
    with Lock(JEDI_MUTEX) as lck:
        try:
            prefix = PREAMBLE + document.stubs() + '\n'
            if _DEBUG:
                print("------------------------------------------------------------------------------------")
                print(prefix)
                print("------------------------------------------------------------------------------------")
                sys.stdout.flush()
            script = jedi.Script(prefix+text, block+prefix.count('\n')+1, column, document.filename)
            items = [(c.name, _try_type(c)) for c in script.completions()
                     if not c.name.startswith('_') and c.name != 'mro']
        except:
            if _DEBUG:
                import traceback
                traceback.print_exc()
            items = None
    return items


def get_docstring(document, text, block, column):
    if jedi is None: return
    from ... import _DEBUG
    with Lock(JEDI_MUTEX) as lck:
        try:
            prefix = PREAMBLE + document.stubs()
            if _DEBUG:
                print("------------------------------------------------------------------------------------")
                print(prefix)
                print("------------------------------------------------------------------------------------")
                sys.stdout.flush()
            script = jedi.Script(prefix+text, block+prefix.count('\n')+1, column, document.filename)
            defs = script.completions()
            if defs:
                doc = defs[0].docstring()
                name = defs[0].name
                if not doc:
                    defs = script.goto_definitions()
                    if defs:
                        name = defs[0].name
                        doc = defs[0].docstring()
                if _DEBUG:
                    d = defs[0]
                    print('{}: [{}] {}'.format(d.name, d.type, d.description))
                    sys.stdout.flush()
                if doc:
                    return name, doc
        except:
            if _DEBUG:
                import traceback
                traceback.print_exc()


def get_definitions(document, text, block, column):
    if jedi is None: return None, None
    with Lock(JEDI_MUTEX) as lck:
        script = jedi.Script(text, block+1, column, document.filename)
        try:
            defs = script.goto_assignments()
        except:
            return None, None
        if defs:
            d = defs[0]
            if d.line is not None:
                return d.line-1, d.column
        return None, None
