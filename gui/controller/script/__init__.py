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
from copy import copy

from ...qt.QtCore import Qt

from ...qt import QtCore, QtGui
from ...utils.qsignals import BlockQtSignals
from ...utils.qthread import BackgroundTask
from .completer import CompletionsController
from ...model.script.completer import get_docstring, get_definitions
from .brackets import get_selections as get_bracket_selections, update_brackets_colors
from .indenter import indent, unindent, autoindent
from ..source import SourceEditController, SourceWidget
from ...model.script import ScriptModel
from ...utils.config import CONFIG, parse_highlight
from ...utils.widgets import EDITOR_FONT
from ...utils.texteditor import TextEditor
from ...external.highlighter import SyntaxHighlighter, load_syntax

if sys.version_info >= (3, 0, 0):
    from ...external.highlighter.python32 import syntax
else:
    from ...external.highlighter.python27 import syntax
from ...external.highlighter.plask import syntax as plask_syntax


syntax['formats'].update(plask_syntax['formats'])
syntax['scanner'][None][-1:-1] = plask_syntax['scanner']


def update_python_scheme():
    global scheme
    scheme = {
        'syntax_comment': parse_highlight(CONFIG['syntax/python_comment']),
        'syntax_string': parse_highlight(CONFIG['syntax/python_string']),
        'syntax_builtin': parse_highlight(CONFIG['syntax/python_builtin']),
        'syntax_keyword': parse_highlight(CONFIG['syntax/python_keyword']),
        'syntax_number': parse_highlight(CONFIG['syntax/python_number']),
        'syntax_member': parse_highlight(CONFIG['syntax/python_member']),
        'syntax_plask': parse_highlight(CONFIG['syntax/python_plask']),
        'syntax_provider': parse_highlight(CONFIG['syntax/python_provider']),
        'syntax_receiver': parse_highlight(CONFIG['syntax/python_receiver']),
        'syntax_log': parse_highlight(CONFIG['syntax/python_log']),
        'syntax_solver': parse_highlight(CONFIG['syntax/python_solver']),
        'syntax_define': parse_highlight(CONFIG['syntax/python_define']),
        'syntax_loaded': parse_highlight(CONFIG['syntax/python_loaded']),
        'syntax_pylab': parse_highlight(CONFIG['syntax/python_pylab']),
    }
update_python_scheme()


class ScriptEditor(TextEditor):
    """Editor with some features usefult for script editing"""

    def __init__(self, parent=None, controller=None):
        self.controller = controller
        super(ScriptEditor, self).__init__(parent)

        self.cursorPositionChanged.connect(self.highlight_brackets)
        self.selectionChanged.connect(self.highlight_brackets)

        self.comment_action = QtGui.QAction('Co&mment lines', self)
        self.uncomment_action = QtGui.QAction('Uncomm&ent lines', self)
        self.comment_action.setShortcut(Qt.CTRL + Qt.Key_Slash)
        self.uncomment_action.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_Slash)
        self.comment_action.triggered.connect(self.block_comment)
        self.uncomment_action.triggered.connect(self.block_uncomment)
        self.addAction(self.comment_action)
        self.addAction(self.uncomment_action)

        self.completer = CompletionsController(self)

        self._pointer_blocked = False
        self._pointer_definition = None, None

        self.setMouseTracking(True)

    def highlight_brackets(self):
        self.setExtraSelections(self.extraSelections() +
                                get_bracket_selections(self, self.textCursor().block(),
                                                       self.textCursor().positionInBlock()))

    def block_comment(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
        if cursor.position() == end: end += 1
        while cursor.position() < end:
            cursor.insertText("# ")
            end += 2
            if not cursor.movePosition(QtGui.QTextCursor.NextBlock):
                break
        cursor.endEditBlock()

    def block_uncomment(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QtGui.QTextCursor.StartOfBlock)
        if cursor.position() == end: end += 1
        document = self.document()
        try:
            while cursor.position() < end:
                while document.characterAt(cursor.position()) in (' ', '\t'):
                    if not cursor.movePosition(QtGui.QTextCursor.NextCharacter): raise ValueError
                if document.characterAt(cursor.position()) == '#':
                    cursor.deleteChar()
                    end -= 1
                    if document.characterAt(cursor.position()) == ' ':
                        cursor.deleteChar()
                        end -= 1
                if not cursor.movePosition(QtGui.QTextCursor.NextBlock): raise ValueError
        except ValueError:
            pass
        cursor.endEditBlock()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if self.completer.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape, Qt.Key_Tab, Qt.Key_Backtab):
                event.ignore()
                return  # let the completer do default behaviors
            elif event.key() == Qt.Key_Backspace:
                self.completer.setCompletionPrefix(self.completer.completionPrefix()[:-1])
            elif event.text():
                last = event.text()[-1]
                if modifiers & ~(Qt.ControlModifier | Qt.ShiftModifier) or \
                        not (last.isalpha() or last.isdigit() or last == '_'):
                    self.completer.popup().hide()
                else:
                    self.completer.setCompletionPrefix(self.completer.completionPrefix() + event.text())
            elif key not in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_AltGr,
                             Qt.Key_Meta, Qt.Key_Super_L, Qt.Key_Super_R):
                self.completer.popup().hide()

        if key in (Qt.Key_Tab, Qt.Key_Backtab) or \
                                key == Qt.Key_Backspace and modifiers != (Qt.ControlModifier | Qt.ShiftModifier):
            cursor = self.textCursor()
            if cursor.hasSelection():
                if key == Qt.Key_Tab:
                    indent(self)
                    return
                elif key == Qt.Key_Backtab:
                    unindent(self)
                    return
            elif key == Qt.Key_Backtab:
                unindent(self)
                return
            else:
                col = cursor.positionInBlock()
                inindent = not cursor.block().text()[:col].strip()
                if inindent:
                    if key == Qt.Key_Tab:
                        indent(self, col)
                        return
                    else:
                        if not (cursor.atBlockStart()):
                            unindent(self, col)
                            return
        elif key == Qt.Key_Home and not modifiers & ~Qt.ShiftModifier:
            cursor = self.textCursor()
            txt = cursor.block().text()
            col = cursor.positionInBlock()
            mode = QtGui.QTextCursor.KeepAnchor if modifiers & Qt.ShiftModifier else QtGui.QTextCursor.MoveAnchor
            if txt[:col].strip() or (col == 0 and txt.strip()):
                cursor.movePosition(QtGui.QTextCursor.StartOfBlock, mode)
                while self.document().characterAt(cursor.position()) in [' ', '\t']:
                    cursor.movePosition(QtGui.QTextCursor.Right, mode)
                self.setTextCursor(cursor)
                return

        if not (key == Qt.Key_Space and modifiers == Qt.ControlModifier) or CONFIG['workarounds/no_jedi']:
            super(ScriptEditor, self).keyPressEvent(event)

        if key in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Colon):
            autoindent(self)
        elif ((key == Qt.Key_Period and not CONFIG['workarounds/jedi_no_dot']) or
              (key == Qt.Key_Space and modifiers == Qt.ControlModifier)) and \
                not self.completer.popup().isVisible() and not CONFIG['workarounds/no_jedi']:
            self.completer.start_completion()

    def link_definition(self, row, col):
        self._pointer_blocked = False
        self._pointer_definition = row, col
        cursor = QtGui.QApplication.overrideCursor()
        if not cursor and row is not None:
            QtGui.QApplication.setOverrideCursor(Qt.PointingHandCursor)
        elif cursor and cursor.shape() == Qt.PointingHandCursor and row is None:
            QtGui.QApplication.restoreOverrideCursor()

    def _get_mouse_definitions(self, event):
        if event.modifiers() == Qt.ControlModifier:
            if self._pointer_blocked: return
            self._pointer_blocked = True
            cursor = self.cursorForPosition(event.pos())
            row = cursor.blockNumber()
            col = cursor.positionInBlock()
            # task = BackgroundTask(lambda: get_definitions(self.controller.document, self.toPlainText(), row, col),
            #                       self.link_definition)
            # task.start()
            if not CONFIG['workarounds/no_jedi']:
                self.link_definition(*get_definitions(self.controller.document, self.toPlainText(), row, col))
        else:
            cursor = QtGui.QApplication.overrideCursor()
            if cursor and cursor.shape() == Qt.PointingHandCursor:
                QtGui.QApplication.restoreOverrideCursor()

    def mouseMoveEvent(self, event):
        super(ScriptEditor, self).mouseMoveEvent(event)
        self._get_mouse_definitions(event)

    def mouseReleaseEvent(self, event):
        super(ScriptEditor, self).mouseReleaseEvent(event)
        row, col = self._pointer_definition
        if event.modifiers() == Qt.ControlModifier and not self._pointer_blocked and row:
            cursor = QtGui.QTextCursor(self.document().findBlockByLineNumber(row))
            cursor.movePosition(QtGui.QTextCursor.Right, QtGui.QTextCursor.MoveAnchor, col)
            self.setTextCursor(cursor)


class ScriptController(SourceEditController):

    def __init__(self, document, model=None):
        if model is None: model = ScriptModel()
        SourceEditController.__init__(self, document, model)
        self.document.window.config_changed.connect(self.reconfig)
        self.highlighter = None

    def create_source_widget(self, parent):
        window = QtGui.QMainWindow(parent)
        window.setWindowFlags(QtCore.Qt.Widget)

        source = SourceWidget(parent, ScriptEditor, self)

        source.editor.setReadOnly(self.model.is_read_only())
        window.editor = source.editor

        self.model.editor = source.editor
        source.editor.cursorPositionChanged.connect(self.model.refresh_info)

        source.toolbar.addSeparator()
        unindent_action = QtGui.QAction(QtGui.QIcon.fromTheme('format-indent-less'), 'Unin&dent', source)
        unindent_action.triggered.connect(lambda: unindent(source.editor))
        source.toolbar.addAction(unindent_action)
        indent_action = QtGui.QAction(QtGui.QIcon.fromTheme('format-indent-more'), '&Indent', source)
        indent_action.triggered.connect(lambda: indent(source.editor))
        source.toolbar.addAction(indent_action)
        menu = QtGui.QMenu()
        menu.addAction(source.editor.comment_action)
        menu.addAction(source.editor.uncomment_action)
        button = QtGui.QToolButton()
        button.setIcon(QtGui.QIcon.fromTheme('document-properties'))
        button.setMenu(menu)
        button.setPopupMode(QtGui.QToolButton.InstantPopup)
        source.toolbar.addWidget(button)
        if self.model.is_read_only():
            unindent_action.setEnabled(False)
            indent_action.setEnabled(False)
            source.editor.comment_action.setEnabled(False)
            source.editor.uncomment_action.setEnabled(False)

        self.help_dock = HelpDock(window)
        parent.config_changed.connect(self.help_dock.reconfig)
        state = CONFIG['session/scriptwindow']
        if state is None or not window.restoreState(state):
            window.addDockWidget(Qt.RightDockWidgetArea, self.help_dock)
        self.help_dock.hide()

        doc_action = QtGui.QAction(QtGui.QIcon.fromTheme('help-contextual'), 'Show &docstring', source)
        doc_action.setShortcut(Qt.SHIFT + Qt.Key_F1)
        doc_action.triggered.connect(self.show_docstring)
        source.editor.addAction(doc_action)
        hide_doc_action = QtGui.QAction('Hide help', source)
        hide_doc_action.setShortcut(Qt.SHIFT + Qt.Key_Escape)
        hide_doc_action.triggered.connect(self.help_dock.hide)
        source.editor.addAction(hide_doc_action)
        source.toolbar.addSeparator()
        source.toolbar.addAction(doc_action)

        self.document.window.closed.connect(self.save_state)

        try:
            loglevel = self.document.loglevel
        except AttributeError:
            pass
        else:
            spacer = QtGui.QWidget()
            spacer.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
            source.toolbar.addWidget(spacer)
            source.toolbar.addWidget(QtGui.QLabel("Log Level: "))
            self.loglevel = QtGui.QComboBox()
            levels = ["Error", "Warning", "Info", "Result", "Data", "Detail", "Debug"]
            self.loglevel.addItems(levels)
            try:
                self.loglevel.setCurrentIndex(levels.index(loglevel.title()))
            except ValueError:
                self.loglevel.setCurrentIndex(5)
            self.loglevel.currentIndexChanged[str].connect(self.document.set_loglevel)
            source.toolbar.addWidget(self.loglevel)

        window.setCentralWidget(source)
        return window

    def save_state(self):
        try:
            CONFIG['session/scriptwindow'] = self.source_widget.saveState()
        except AttributeError:
            pass
        else:
            CONFIG.sync()

    def on_edit_enter(self):
        self.rehighlight()
        super(ScriptController, self).on_edit_enter()

    def rehighlight(self):
        if (self.document.solvers and self.document.solvers.model.entries) or \
                (self.document.defines and self.document.defines.model.entries):
            current_syntax = {'formats': syntax['formats'],
                              'partitions': syntax['partitions'],
                              'scanner': copy(syntax['scanner'])}
            current_syntax['scanner'][None] = copy(syntax['scanner'][None])
            defines = [e.name for e in self.document.defines.model.entries]
            if defines:
                current_syntax['scanner'][None].insert(0, ('define', defines, '(^|[^\\.\\w])', '[\x08\\W]'))
            solvers = [e.name for e in self.document.solvers.model.entries]
            if solvers:
                current_syntax['scanner'][None].insert(0, ('solver', solvers, '(^|[^\\.\\w])', '[\x08\\W]'))
        else:
            current_syntax = syntax
        self.highlighter = SyntaxHighlighter(self.source_widget.editor.document(),
                                             *load_syntax(current_syntax, scheme),
                                             default_font=EDITOR_FONT)
        self.highlighter.rehighlight()

    def reconfig(self):
        editor = self.source_widget.editor
        editor.setFont(EDITOR_FONT)
        if editor.line_numbers is not None:
            editor.line_numbers.setFont(EDITOR_FONT)
        update_brackets_colors()
        if self.highlighter is not None:
            with BlockQtSignals(editor):
                update_python_scheme()
                self.rehighlight()

    def on_edit_exit(self):
        return super(ScriptController, self).on_edit_exit()

    def show_docstring(self):
        if CONFIG['workarounds/no_jedi']: return
        cursor = self.source_widget.editor.textCursor()
        cursor.movePosition(QtGui.QTextCursor.EndOfWord)
        row = cursor.blockNumber()
        col = cursor.positionInBlock()
        # QtGui.QApplication.setOverrideCursor(Qt.BusyCursor)
        if CONFIG['workarounds/blocking_jedi']:
            self.help_dock.show_help(get_docstring(self.document, self.source_widget.editor.toPlainText(), row, col))
        else:
            task = BackgroundTask(lambda: get_docstring(self.document, self.source_widget.editor.toPlainText(),
                                                        row, col),
                                  self.help_dock.show_help)
            task.start()


class HelpDock(QtGui.QDockWidget):

    def __init__(self, parent):
        super(HelpDock, self).__init__(parent)
        self.textarea = QtGui.QTextEdit()
        self.textarea.setReadOnly(True)
        help_font = QtGui.QFont(EDITOR_FONT)
        help_font.fromString(','.join(CONFIG['editor/help_font']))
        pal = self.textarea.palette()
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffe"))
        self.textarea.setPalette(pal)
        self.textarea.setFont(help_font)
        font_metrics = self.textarea.fontMetrics()
        self.textarea.setMinimumWidth(86 * font_metrics.width('a'))
        self.textarea.setMinimumHeight(8 * font_metrics.height())
        self.setWidget(self.textarea)
        self.setObjectName('help')

    def show_help(self, name=None, docstring=None):
        if docstring is not None:
            self.setWindowTitle("Help: " + name)
            self.textarea.setText(docstring)
            self.show()
        # QtGui.QApplication.restoreOverrideCursor()

    def reconfig(self):
        help_font = self.textarea.font()
        help_font.fromString(','.join(CONFIG['editor/help_font']))
        self.textarea.setFont(help_font)
        font_metrics = self.textarea.fontMetrics()
        self.textarea.setMinimumWidth(86 * font_metrics.width('a'))
        self.textarea.setMinimumHeight(8 * font_metrics.height())