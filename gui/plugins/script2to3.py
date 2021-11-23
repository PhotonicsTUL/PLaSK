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

# plugin: Convert Script to Python 3
# description: Show a tool to convert current script to Python 3 for use with future/current versions of PLaSK.

try:
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package
except ImportError:
    pass
else:
    refactoring_tool = RefactoringTool(fixer_names=get_fixers_from_package('lib2to3.fixes'))

    import gui
    from gui.qt.QtWidgets import *
    from gui.qt.QtGui import *
    from gui.qt import qt_exec
    from gui.utils.config import CONFIG

    class RefactorAction(QAction):

        def __init__(self, parent):
            super().__init__(QIcon.fromTheme('show-source'), 'Convert Script to Python&3', parent)
            CONFIG.set_shortcut(self, 'convert_to_python3', 'Convert Script to Python3')
            self.triggered.connect(self.execute)

        def execute(self):
            tabs = self.parent().tabs
            tabs.setCurrentIndex(tabs.count()-1)
            editor = self.parent().document.script.get_source_widget().editor
            text2 = editor.toPlainText() + "\n"
            try:
                node3 = refactoring_tool.refactor_string(text2, 'script')
            except Exception as err:
                msgbox = QMessageBox()
                msgbox.setWindowTitle("Python3 Conversion Error")
                msgbox.setText("There was an error while converting your script to Python3!")
                msgbox.setInformativeText("Check if the script does not have any Python2 syntax errors.")
                msgbox.setDetailedText(str(err))
                msgbox.setStandardButtons(QMessageBox.StandardButton.Ok)
                msgbox.setIcon(QMessageBox.Icon.Critical)
                qt_exec(msgbox)
            else:
                editor.moveCursor(QTextCursor.MoveOperation.Start)
                try:
                    editor.moveCursor(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
                except (TypeError, AttributeError):
                    editor.moveCursor(QTextCursor.MoveOperation.End, True)
                cursor = editor.textCursor()
                cursor.insertText(str(node3)[:-1])

                msgbox = QMessageBox()
                msgbox.setWindowTitle("Python3 Conversion")
                msgbox.setText("Your script have been converted to Python3.")
                msgbox.setInformativeText("You should verify it now and do the manual adjustments if necessary.")
                msgbox.setStandardButtons(QMessageBox.StandardButton.Ok)
                msgbox.setIcon(QMessageBox.Icon.Information)
                qt_exec(msgbox)

    if gui.ACTIONS:
        gui.ACTIONS.append(None)
    gui.ACTIONS.append(RefactorAction)
