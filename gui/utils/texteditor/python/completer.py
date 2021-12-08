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
import os

from ....qt.QtCore import *
from ....qt.QtGui import *
from ....qt.QtWidgets import *

from ....utils.qthread import BackgroundTask, Lock
from ....utils.config import CONFIG

try:
    import jedi
except ImportError:
    jedi_path = os.path.join(__path__[0], 'lib', 'jedi')
    if jedi_path not in sys.path:
        sys.path.insert(1, jedi_path)
        import jedi
    else:
        jedi = None

JEDI_MUTEX = QMutex()


PREAMBLE = """\
from pylab import *
import plask
from plask import *
import plask.geometry, plask.mesh, plask.material, plask.flow, plask.phys, plask.algorithm
from plask import geometry, mesh, material, flow, phys, algorithm
from plask.pylab import *
from plask.hdf5 import *
"""


def preload_jedi_modules():
    with Lock(JEDI_MUTEX):
        try:
            jedi.Script(PREAMBLE, 8, 0, None).completions()
        except:
            from ... import _DEBUG
            if _DEBUG:
                import traceback
                traceback.print_exc()


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


class CompletionsModel(QAbstractTableModel):

    _icons = {}

    def __init__(self, items):
        super().__init__()
        self.items = items

        if not self._icons:
            self._load_icons()

    def _load_icons(self):
        code_function = QIcon.fromTheme("code-function")
        code_class = QIcon.fromTheme("code-class")
        code_variable = QIcon.fromTheme("code-variable")
        code_typedef = QIcon.fromTheme("code-typedef")
        code_block = QIcon.fromTheme("code-block")
        code_context = QIcon.fromTheme("code-context")
        no_icon = QIcon()
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

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        return ('icon', 'completion')[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            row = index.row()
            col = index.column()
            value = self.items[row]
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                return value[0]
            elif role == Qt.ItemDataRole.DecorationRole:
                return self._icons.get(value[1], QIcon())
            else:
                return None
        else:
            return None

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def columnCount(self, parent=QModelIndex()):
        return 1


def _try_type(compl):
    try:
        return compl.type
    except:
        return None


def get_completions(document, text, block, column):
    if jedi is None or CONFIG['workarounds/no_jedi']: return
    from .... import _DEBUG
    with Lock(JEDI_MUTEX) as lck:
        try:
            prefix = PREAMBLE + document.stubs() + '\n'
            if _DEBUG:
                print("------------------------------------------------------------------------------------\n" +
                      prefix + "\n------------------------------------------------------------------------------------",
                      file=sys.stderr)
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
    if jedi is None or CONFIG['workarounds/no_jedi']: return
    from .... import _DEBUG
    with Lock(JEDI_MUTEX) as lck:
        try:
            prefix = PREAMBLE + document.stubs()
            if _DEBUG:
                print("------------------------------------------------------------------------------------\n" +
                      prefix + "\n------------------------------------------------------------------------------------",
                      file=sys.stderr)
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
                    print('{}: [{}] {} "{}..."'.format(d.name, d.type, d.description, doc[:8]), file=sys.stderr)
                if doc:
                    return name, doc
                else:
                    return name, ''
        except:
            if _DEBUG:
                import traceback
                traceback.print_exc()


def get_definitions(document, text, block, column):
    if jedi is None or CONFIG['workarounds/no_jedi']: return None, None
    with Lock(JEDI_MUTEX) as lck:
        try:
            script = jedi.Script(text, block+1, column, document.filename)
            defs = script.goto_assignments()
        except:
            return None, None
        if defs:
            d = defs[0]
            if d.line is not None:
                return d.line-1, d.column
        return None, None


class CompletionsController(QCompleter):

    def __init__(self, edit, parent=None):
        super().__init__(parent)
        self._edit = edit
        self.setWidget(edit)
        self.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
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
            cursor.movePosition(QTextCursor.MoveOperation.Left)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfWord)
        cursor.insertText(completion[extra:])
        self._edit.setTextCursor(cursor)

    def start_completion(self):
        if CONFIG['workarounds/no_jedi']: return

        cursor = self._edit.textCursor()
        row = cursor.blockNumber()
        col = cursor.positionInBlock()
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        completion_prefix = cursor.selectedText()

        def thread_finished(completions):
            tc = self._edit.textCursor()
            if tc.blockNumber() == row and tc.positionInBlock() == col:
                self.show_completion_popup(completion_prefix, completions)
            # QApplication.restoreOverrideCursor()

        # QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        if CONFIG['workarounds/blocking_jedi']:
            thread_finished(self._edit.get_completions(row, col))
        else:
            task = BackgroundTask(lambda: self._edit.get_completions(row, col), thread_finished)
            task.start()

    def show_completion_popup(self, completion_prefix, completions):
        if completions is not None:
            self.setModel(CompletionsModel(completions))
            if completion_prefix != self.completionPrefix():
                self.setCompletionPrefix(completion_prefix)
                self.popup().setCurrentIndex(self.completionModel().index(0, 0))
            rect = self._edit.cursorRect()
            rect.setWidth(self.popup().sizeHintForColumn(0) + self.popup().verticalScrollBar().sizeHint().width())
            self.complete(rect)  # popup it up!
