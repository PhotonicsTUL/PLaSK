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

# plugin: Debugger
# description: Add a simple debugger for local computations.
import os

from gui.controller.script import ScriptController
from gui.launch import LAUNCHERS
from gui.launch.dock import OutputWindow
from gui.launch.local import Launcher as LocalLauncher
from gui.launch.local import LaunchThread
from gui.qt import QtSignal
from gui.qt.QtCore import *
from gui.qt.QtGui import *
from gui.qt.QtWidgets import *
from gui.utils.config import CONFIG
from gui.utils.files import which
from gui.utils.texteditor import LINENUMBER_BACKGROUND_COLOR, LINENUMBER_FOREGROUND_COLOR, EditorWidget, LineNumberArea
from gui.utils.texteditor.python import PythonEditor, PythonEditorWidget

from .ui import DebuggerPanel


class DebuggerLineNumberArea(LineNumberArea):

    def __init__(self, editor):
        super().__init__(editor)
        self.breakpoints = set()

    def get_width(self):
        width = super().get_width()
        if self.breakpoints:
            width += 12  # Extra space for breakpoint marker
        return width

    def get_breakpoints(self):
        return sorted(self.breakpoints)

    def update_width(self):
        cr = self.editor.contentsRect()
        self.setGeometry(QRect(cr.left(), cr.top(), self.get_width(), cr.height()))
        super().update_width()

    def mousePressEvent(self, event):
        y = event.pos().y()
        block = self.editor.firstVisibleBlock()
        top = (self.editor.blockBoundingGeometry(block).translated(self.editor.contentOffset()).top())
        bottom = top + self.editor.blockBoundingRect(block).height()
        while block.isValid() and top <= y:
            if block.isVisible() and bottom >= y:
                line_number = block.blockNumber() + 1 + self._offset
                if line_number in self.breakpoints:
                    self.breakpoints.remove(line_number)
                else:
                    self.breakpoints.add(line_number)
                self.update()
                break
            block = block.next()
            top = bottom
            bottom = top + self.editor.blockBoundingRect(block).height()
        self.update_width()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), LINENUMBER_BACKGROUND_COLOR)
        block = self.editor.firstVisibleBlock()
        block_number = block.blockNumber() + 1 + self._offset
        top = (self.editor.blockBoundingGeometry(block).translated(self.editor.contentOffset()).top())
        bottom = top + self.editor.blockBoundingRect(block).height()
        width = self.width()
        if self.breakpoints:
            width -= 12  # Adjust width for breakpoint marker
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(LINENUMBER_FOREGROUND_COLOR)
                painter.drawText(
                    0,
                    int(top),
                    width - 3,
                    self.editor.fontMetrics().height(),
                    Qt.AlignmentFlag.AlignRight,
                    str(block_number),
                )
                # Draw breakpoint marker
                if block_number in self.breakpoints:
                    radius = 5
                    center_x = width + 5
                    center_y = int(top + self.editor.fontMetrics().height() / 2)
                    painter.setBrush(QColor('red'))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(QPoint(center_x, center_y), radius, radius)
            block = block.next()
            top = bottom
            bottom = top + self.editor.blockBoundingRect(block).height()
            block_number += 1


class DebuggerPythonEditor(PythonEditor):

    LineNumbers = DebuggerLineNumberArea

    breakpoints_ready = QtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_lines = set()

    def update_current_debug_line(self, line):
        self.debug_lines = set([line])
        self.update_selections()

    def update_selections(self, selections=None):
        """Add our own custom selections"""
        if selections is not None:
            self.selections = selections
        self.setExtraSelections(
            self.highlight_current_line() + self.get_same_as_selected() + self.get_debugger_selections() + self.selections
        )

    def get_debugger_selections(self):
        if not self.debug_lines:
            return []

        doc = self.document()
        # fm = self.fontMetrics()
        selections = []

        for line_no in self.debug_lines:
            block = doc.findBlockByNumber(line_no - 1)
            if not block.isValid():
                continue
            cursor = QTextCursor(block)

            sel = QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.cursor.clearSelection()

            sel.format.setBackground(QColor("#4444aa80"))
            sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)

            selections.append(sel)

        return selections

    def send_breakpoints(self):
        self.breakpoints_ready.emit(self.line_numbers.get_breakpoints())


class DebuggerPythonEditorWidget(PythonEditorWidget):

    Editor = DebuggerPythonEditor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugger = DebuggerPanel(self)

        self._editor_widget.editor.breakpoints_ready.connect(self.debugger.recieve_breakpoints)
        self.debugger.ask_breakpoints.connect(self.editor.send_breakpoints)
        self.debugger.current_line_signal.connect(self.editor.update_current_debug_line)

        self._editor_widget.add_action("&Open debugger", "debugger-open", None, self.debugger.toggle_visibility)


class Launcher(LocalLauncher):
    name = "Local Debugger"

    def launch(self, main_window, args, defs):
        program = CONFIG['launcher_local/program']
        if not (program and os.path.isfile(program) and os.access(program, os.X_OK)):
            program = 'plask'
            if os.name == 'nt':
                program += '.exe'
            program = which(program) or program

        if main_window.isWindowModified():
            confirm = QMessageBox.question(
                main_window, "Unsaved File", "The file must be saved before launching local computations. "
                "Do you want to save the file now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if confirm == QMessageBox.StandardButton.No or not main_window.save():
                return

        filename = os.path.abspath(main_window.document.filename)

        if self.dirname:
            dirname = self.dirname
        else:
            dirname = os.path.dirname(filename)

        dock = OutputWindow(self, main_window)

        try:
            bottom_docked = [
                w for w in main_window.findChildren(QDockWidget)
                if main_window.dockWidgetArea(w) == Qt.DockWidgetArea.BottomDockWidgetArea
            ][-1]
        except IndexError:
            main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        else:
            main_window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            main_window.tabifyDockWidget(bottom_docked, dock)
            dock.show()
            dock.raise_()

        debugger = main_window.document.controller_by_name('script').get_source_widget().debugger

        breakpoints = ",".join([f"{filename}:" + str(bp) for bp in debugger.get_breakpoint_lines()])
        port = CONFIG['launcher_debug/port']
        debugger_path = os.path.join(os.path.dirname(__file__), 'run.py')

        dbs_args = ['--breakpoints', breakpoints, '--port', str(port)] + (list(defs) if defs else []) + \
                   ['--', filename] + list(args)

        dock.thread = LaunchThread(program, debugger_path, dirname, dock, main_window, dbs_args, [])

        dock.thread.finished.connect(dock.thread_finished)
        dock.thread.start()

        debugger.show()
        debugger.connect_debugger()


# Initialize the plugin

ScriptController.SourceWidget = DebuggerPythonEditorWidget
LAUNCHERS['local_dbg'] = Launcher()
