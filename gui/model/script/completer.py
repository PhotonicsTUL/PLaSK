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

from ... import _DEBUG

try:
    import jedi
except ImportError:
    jedi = None


JEDI_LOCK = QtCore.QMutex()


class _JediThread(QtCore.QThread):

    def __init__(self):
        super(_JediThread, self).__init__()
        self.finished.connect(_JediThread.delete_global)

    def run(self):
        JEDI_LOCK.lock()
        try:
            jedi.preload_module('pylab', 'plask')
        finally:
            JEDI_LOCK.unlock()

    @staticmethod
    def delete_global():
        global _jedi_thread
        try:
            del _jedi_thread
        except NameError:
            pass


def prepare_completions():
    if jedi:
        # if _DEBUG:
        #     def print_debug(obj, txt):
        #         sys.stderr.write(txt + '\n')
        #     jedi.set_debug_function(print_debug)
        global _jedi_thread
        _jedi_thread = _JediThread()
        _jedi_thread.start()


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
                            "forflow": code_context,
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


PREAMBLE = '''
from pylab import *
from plask import *
from plask import geometry, mesh, material, flow, phys, algorithm
import plask.geometry, plask.mesh, plask.material, plask.plow, plask.phys, plask.algorithm
from plask.pylab import *
from plask.hdf5 import *
'''


def _try_type(compl):
    try:
        return compl.type
    except:
        return None


def get_completions(document, text, block, column):
    if jedi is None: return
    JEDI_LOCK.lock()
    try:
        prefix = PREAMBLE + document.stubs()
        script = jedi.Script(prefix+text, block+prefix.count('\n')+1, column, document.filename)
        items = [(c.name, _try_type(c)) for c in script.completions() if not c.name.startswith('_') and c.name != 'mro']
    except:
        items = None
    finally:
        JEDI_LOCK.unlock()
    return items


def get_docstring(document, text, block, column):
    if jedi is None: return
    JEDI_LOCK.lock()
    try:
        prefix = PREAMBLE + document.stubs()
        script = jedi.Script(prefix+text, block+prefix.count('\n')+1, column, document.filename)
        defs = script.goto_definitions()
        if defs:
            return defs[0].name, defs[0].docstring()
    except:
        pass
    finally:
        JEDI_LOCK.unlock()
