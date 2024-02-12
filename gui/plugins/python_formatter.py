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

# Copyright (C) 2022 Photonics Group, Lodz University of Technology
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

# plugin: Python Formatter
# description: Format Python code using configured formatter

import sys
import shutil
from subprocess import Popen, PIPE

import gui
from gui.qt.QtCore import Qt
from gui.qt.QtWidgets import *
from gui.qt.QtGui import *
from gui.qt import qt_exec
from gui.utils.config import CONFIG
from gui.utils.config import DEFAULTS as CONFIG_DEFAULTS
from gui.utils.settings import CONFIG_WIDGETS, Path, MultiLine

yapf = shutil.which('yapf')
CONFIG_DEFAULTS['python_formatter/program'] = yapf if yapf is not None else 'yapf'
CONFIG_DEFAULTS['python_formatter/args'] = '--style={based_on_style: pep8, column_limit: 132, dedent_closing_brackets: true, ' \
                                           'coalesce_brackets: true}'


class FormatPythonAction(QAction):

    def __init__(self, parent):
        super().__init__(QIcon.fromTheme('document-edit'), 'Format P&ython Script', parent)
        CONFIG.set_shortcut(self, 'format_python', 'Format P&ython Script', Qt.Modifier.CTRL | Qt.Modifier.SHIFT | Qt.Key.Key_I)
        self.triggered.connect(self.execute)

    def execute(self):
        tabs = self.parent().tabs
        tabs.setCurrentIndex(tabs.count() - 1)
        editor = self.parent().document.script.get_source_widget().editor
        offset = editor.line_numbers.offset
        text = ''.join(f"# {i}\n" for i in range(offset)) + editor.toPlainText() + "\n"
        try:
            program = CONFIG['python_formatter/program']
            args = CONFIG['python_formatter/args'].splitlines()
            proc = Popen([program] + args, stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='UTF-8')
            text, errors = proc.communicate(text)
            if proc.returncode != 0:
                raise RuntimeError(errors)
        except Exception as err:
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Python Formatting Error")
            msgbox.setText("There was an error while formatting your script!")
            msgbox.setDetailedText(str(err))
            msgbox.setStandardButtons(QMessageBox.StandardButton.Ok)
            msgbox.setIcon(QMessageBox.Icon.Critical)
            qt_exec(msgbox)
        else:
            lineno = editor.textCursor().blockNumber()
            scroll = editor.verticalScrollBar().value()
            editor.moveCursor(QTextCursor.MoveOperation.Start)
            try:
                editor.moveCursor(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
            except (TypeError, AttributeError):
                editor.moveCursor(QTextCursor.MoveOperation.End, True)
            cursor = editor.textCursor()
            cursor.insertText('\n'.join(text.splitlines()[offset:]))
            cur = QTextCursor(editor.document().findBlockByLineNumber(min(lineno, editor.document().blockCount() - 1)))
            editor.setTextCursor(cur)
            editor.verticalScrollBar().setValue(min(scroll, editor.verticalScrollBar().maximum()))
            editor.setFocus()


CONFIG_WIDGETS["Editor"]["Python Formatter"] = [
    ("Formatter", Path('python_formatter/program', "Python formatter executable",
                        "Executable (*{0})".format('.exe' if sys.platform == 'win32' else ''),
                        "Full patch to Python formatter executable.")),
    ("Arguments", MultiLine('python_formatter/args', "Arguments for Python formatter. "
                                                        "Each argument is a separate item in the array.")),
]

if gui.ACTIONS:
    gui.ACTIONS.append(None)
gui.ACTIONS.append(FormatPythonAction)
