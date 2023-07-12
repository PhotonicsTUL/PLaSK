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


import weakref

from ....qt import QT_API
from ....qt.QtCore import *
from ....qt.QtWidgets import *
from ....qt.QtGui import *
from ....utils.qsignals import BlockQtSignals
from ....utils.qthread import BackgroundTask
from ....utils.config import CONFIG, parse_highlight, set_font
from ....utils.widgets import EDITOR_FONT
from ....utils.help import open_help
from ....lib.highlighter import SyntaxHighlighter, load_syntax
from ....lib.highlighter.plask import get_syntax
from .. import TextEditor, EditorWidget
from .brackets import get_selections as get_bracket_selections, update_brackets_colors
from .indenter import indent, unindent, autoindent
from .completer import CompletionsController, get_completions, get_docstring, get_definitions

from numpy import inf


PYTHON_SCHEME = {}


def update_python_scheme():
    global PYTHON_SCHEME
    PYTHON_SCHEME['syntax_comment'] = parse_highlight(CONFIG['syntax/python_comment'])
    PYTHON_SCHEME['syntax_string'] = parse_highlight(CONFIG['syntax/python_string'])
    PYTHON_SCHEME['syntax_special'] = parse_highlight(CONFIG['syntax/python_special'])
    PYTHON_SCHEME['syntax_builtin'] = parse_highlight(CONFIG['syntax/python_builtin'])
    PYTHON_SCHEME['syntax_keyword'] = parse_highlight(CONFIG['syntax/python_keyword'])
    PYTHON_SCHEME['syntax_number'] = parse_highlight(CONFIG['syntax/python_number'])
    PYTHON_SCHEME['syntax_decorator'] = parse_highlight(CONFIG['syntax/python_decorator'])
    PYTHON_SCHEME['syntax_member'] = parse_highlight(CONFIG['syntax/python_member'])
    PYTHON_SCHEME['syntax_plask'] = parse_highlight(CONFIG['syntax/python_plask'])
    PYTHON_SCHEME['syntax_provider'] = parse_highlight(CONFIG['syntax/python_provider'])
    PYTHON_SCHEME['syntax_receiver'] = parse_highlight(CONFIG['syntax/python_receiver'])
    PYTHON_SCHEME['syntax_log'] = parse_highlight(CONFIG['syntax/python_log'])
    PYTHON_SCHEME['syntax_solver'] = parse_highlight(CONFIG['syntax/python_solver'])
    PYTHON_SCHEME['syntax_define'] = parse_highlight(CONFIG['syntax/python_define'])
    PYTHON_SCHEME['syntax_loaded'] = parse_highlight(CONFIG['syntax/python_loaded'])
    PYTHON_SCHEME['syntax_pylab'] = parse_highlight(CONFIG['syntax/python_pylab'])
    PYTHON_SCHEME['syntax_obsolete'] = {'color': '#aaaaaa', 'bold': True, 'italic': True}
update_python_scheme()


class PythonEditor(TextEditor):
    """Editor with some features usefult for script editing"""

    def __init__(self, parent=None, document=None, line_numbers=True):
        super().__init__(parent, line_numbers)

        self.cursorPositionChanged.connect(self.highlight_brackets)
        self.selectionChanged.connect(self.highlight_brackets)

        self.comment_action = QAction('Co&mment Lines', self)
        self.uncomment_action = QAction('Uncomm&ent Lines', self)
        self.toggle_comment_action = QAction('&Toggle Comment on Lines', self)
        self.join_lines_action = QAction('Join Lines', self)
        CONFIG.set_shortcut(self.comment_action, 'python_comment')
        CONFIG.set_shortcut(self.uncomment_action, 'python_uncomment')
        CONFIG.set_shortcut(self.toggle_comment_action, 'python_toggle_comment')
        CONFIG.set_shortcut(self.join_lines_action, 'python_join_lines')
        self.comment_action.triggered.connect(self.block_comment)
        self.uncomment_action.triggered.connect(self.block_uncomment)
        self.toggle_comment_action.triggered.connect(self.block_comment_toggle)
        self.join_lines_action.triggered.connect(self.join_lines)
        self.addAction(self.comment_action)
        self.addAction(self.uncomment_action)
        self.addAction(self.toggle_comment_action)
        self.addAction(self.join_lines_action)

        self.complete_action = QAction('Show Completer', self)
        CONFIG.set_shortcut(self.complete_action, 'python_completion')
        self.complete_action.triggered.connect(self.show_completer)
        self.addAction(self.complete_action)

        self.completer = CompletionsController(self)

        self._pointer_blocked = False
        self._pointer_definition = None, None

        self.setMouseTracking(True)

        if isinstance(document, weakref.ProxyType):
            self._document = document
        elif document is not None:
            self._document = weakref.proxy(document)
        else:
            self._document = None

        self.highlighter = None

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
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        start = cursor.position()
        if start == end: end += 1
        document = self.document()
        margin = inf
        while cursor.position() < end:
            while document.characterAt(cursor.position()) in (' ', '\t'):
                if not cursor.movePosition(QTextCursor.MoveOperation.NextCharacter): break
            margin = min(cursor.positionInBlock(), margin)
            if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock):
                break
        cursor.setPosition(start)
        while cursor.position() < end:
            cursor.movePosition(QTextCursor.MoveOperation.NextCharacter, QTextCursor.MoveMode.MoveAnchor, margin)
            cursor.insertText("# ")
            end += 2
            if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock):
                break
        cursor.endEditBlock()

    def block_uncomment(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        if cursor.position() == end: end += 1
        document = self.document()
        try:
            while cursor.position() < end:
                while document.characterAt(cursor.position()) in (' ', '\t'):
                    if not cursor.movePosition(QTextCursor.MoveOperation.NextCharacter): raise ValueError
                if document.characterAt(cursor.position()) == '#':
                    cursor.deleteChar()
                    end -= 1
                    if document.characterAt(cursor.position()) == ' ':
                        cursor.deleteChar()
                        end -= 1
                if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock): raise ValueError
        except ValueError:
            pass
        cursor.endEditBlock()

    def block_comment_toggle(self):
        incomment = False
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        if cursor.position() == end: end += 1
        document = self.document()
        try:
            while cursor.position() < end:
                while document.characterAt(cursor.position()) in (' ', '\t'):
                    if not cursor.movePosition(QTextCursor.MoveOperation.NextCharacter): raise ValueError
                if document.characterAt(cursor.position()) == '#':
                    incomment = True
                elif not cursor.atBlockEnd():
                    incomment = False
                    break
                if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock): raise ValueError
        except ValueError:
            pass
        if incomment:
            self.block_uncomment()
        else:
            self.block_comment()

    def join_lines(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        if cursor.atEnd(): return
        document = self.document()
        cursor.beginEditBlock()
        cursor.deleteChar()
        while document.characterAt(cursor.position()) in ' \t':
            cursor.deleteChar()
        cursor.insertText(' ')
        cursor.endEditBlock()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if self.completer.popup().isVisible():
            if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Escape, Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                event.ignore()
                return  # let the completer do default behaviors
            elif event.key() == Qt.Key.Key_Backspace:
                self.completer.setCompletionPrefix(self.completer.completionPrefix()[:-1])
            elif event.text():
                last = event.text()[-1]
                if modifiers & ~(Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier) or \
                        not (last.isalpha() or last.isdigit() or last == '_'):
                    self.completer.popup().hide()
                else:
                    self.completer.setCompletionPrefix(self.completer.completionPrefix() + event.text())
            elif key not in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_AltGr,
                             Qt.Key.Key_Meta, Qt.Key.Key_Super_L, Qt.Key.Key_Super_R):
                self.completer.popup().hide()

        if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab) or \
                key == Qt.Key.Key_Backspace and modifiers != (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            cursor = self.textCursor()
            if cursor.hasSelection():
                if key == Qt.Key.Key_Tab:
                    indent(self)
                    return
                elif key == Qt.Key.Key_Backtab:
                    unindent(self)
                    return
            elif key == Qt.Key.Key_Backtab:
                unindent(self)
                return
            else:
                col = cursor.positionInBlock()
                inindent = not cursor.block().text()[:col].strip()
                if inindent:
                    if key == Qt.Key.Key_Tab:
                        indent(self, col)
                        return
                    else:
                        if not (cursor.atBlockStart()):
                            unindent(self, col)
                            return
        elif key == Qt.Key.Key_Home and not modifiers & ~Qt.KeyboardModifier.ShiftModifier:
            cursor = self.textCursor()
            txt = cursor.block().text()
            col = cursor.positionInBlock()
            mode = QTextCursor.MoveMode.KeepAnchor if modifiers & Qt.KeyboardModifier.ShiftModifier else QTextCursor.MoveMode.MoveAnchor
            if txt[:col].strip() or (col == 0 and txt.strip()):
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, mode)
                while self.document().characterAt(cursor.position()) in [' ', '\t']:
                    cursor.movePosition(QTextCursor.MoveOperation.Right, mode)
                self.setTextCursor(cursor)
                return

        super().keyPressEvent(event)

        if key in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Colon):
            autoindent(self)
        elif key == Qt.Key.Key_Period and not (CONFIG['workarounds/no_jedi'] or CONFIG['workarounds/jedi_no_dot'] or
                                           self.completer.popup().isVisible()):
            self.completer.start_completion()

    def rehighlight(self, *args, **kwargs):
        syntax = get_syntax(*args, **kwargs)
        self.highlighter = SyntaxHighlighter(self.document(), *load_syntax(syntax, PYTHON_SCHEME), default_font=EDITOR_FONT)
        self.highlighter.rehighlight()

    def reconfig(self, *args, **kwargs):
        super().reconfig()
        update_brackets_colors()
        if self.highlighter is not None:
            with BlockQtSignals(self):
                update_python_scheme()
                self.rehighlight(*args, **kwargs)

    def link_definition(self, row, col):
        self._pointer_blocked = False
        self._pointer_definition = row, col
        cursor = QApplication.overrideCursor()
        if not cursor and row is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)
        elif cursor and cursor.shape() == Qt.CursorShape.PointingHandCursor and row is None:
            QApplication.restoreOverrideCursor()

    def _get_mouse_definitions(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if self._pointer_blocked: return
            self._pointer_blocked = True
            cursor = self.cursorForPosition(event.pos())
            row = cursor.blockNumber()
            col = cursor.positionInBlock()
            # task = BackgroundTask(lambda: get_definitions(self._document, self.toPlainText(), row, col),
            #                       self.link_definition)
            # task.start()
            if not CONFIG['workarounds/no_jedi']:
                self.link_definition(*get_definitions(self._document, self.toPlainText(), row, col))
        else:
            cursor = QApplication.overrideCursor()
            if cursor and cursor.shape() == Qt.CursorShape.PointingHandCursor:
                QApplication.restoreOverrideCursor()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self._get_mouse_definitions(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        row, col = self._pointer_definition
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and not self._pointer_blocked and row:
            cursor = QTextCursor(self.document().findBlockByLineNumber(row))
            cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor, col)
            self.setTextCursor(cursor)

    def get_completions(self, row, col):
        if self._document is not None:
            return get_completions(self._document, self.toPlainText(), row, col)
        else:
            return ()

    def show_completer(self):
        if not (CONFIG['workarounds/no_jedi'] or self.completer.popup().isVisible()):
            self.completer.start_completion()

    def remove_trailing_spaces(self):
        if CONFIG['editor/remove_trailing_spaces']:
            document = self.document()
            cursor = self.textCursor()
            cursor.beginEditBlock()
            regex = QRegularExpression(r'\s+$')
            found = document.find(regex)
            while found and not found.isNull():
                found.removeSelectedText()
                found = document.find(regex, found.position())
            cursor.endEditBlock()


class PythonEditorWidget(QMainWindow):

    def __init__(self, parent=None, document=None, read_only=False, *args, **kwargs):
        super().__init__(parent)

        self.setWindowFlags(Qt.WindowType.Widget)

        widget = EditorWidget(parent, PythonEditor, document, *args, **kwargs)
        self.editor = widget.editor

        widget.toolbar.addSeparator()
        unindent_action = QAction(QIcon.fromTheme('format-indent-less'), 'Unin&dent', widget)
        unindent_action.triggered.connect(lambda: unindent(widget.editor))
        widget.toolbar.addAction(unindent_action)
        indent_action = QAction(QIcon.fromTheme('format-indent-more'), '&Indent', widget)
        indent_action.triggered.connect(lambda: indent(widget.editor))
        widget.toolbar.addAction(indent_action)
        button = QToolButton()
        button.setIcon(QIcon.fromTheme('document-properties'))
        menu = QMenu(button)
        menu.addAction(widget.editor.comment_action)
        menu.addAction(widget.editor.uncomment_action)
        menu.addAction(widget.editor.toggle_comment_action)
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        widget.toolbar.addWidget(button)

        if read_only:
            widget.editor.setReadOnly(True)
            unindent_action.setEnabled(False)
            indent_action.setEnabled(False)
            widget.editor.comment_action.setEnabled(False)
            widget.editor.uncomment_action.setEnabled(False)
            widget.editor.toggle_comment_action.setEnabled(False)

        if document is not None:
            if isinstance(document, weakref.ProxyType):
                self._document = document
            else:
                self._document = weakref.proxy(document)

            self.help_dock = HelpDock(self)
            parent.config_changed.connect(self.help_dock.reconfig)
            state = CONFIG['session/scriptwindow']
            if state is None or not self.restoreState(state):
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.help_dock)
            self.help_dock.hide()

            widget.toolbar.addSeparator()

            help_action = QAction(QIcon.fromTheme('help-contents'), 'Open &Help', widget)
            CONFIG.set_shortcut(help_action, 'python_help')
            help_action.triggered.connect(self.open_help)
            widget.editor.addAction(help_action)
            widget.toolbar.addAction(help_action)

            doc_action = QAction(QIcon.fromTheme('help-contextual'), 'Show &Docstring', widget)
            CONFIG.set_shortcut(doc_action, 'python_docstring')
            doc_action.triggered.connect(self.show_docstring)
            widget.editor.addAction(doc_action)
            hide_doc_action = QAction('Hide Docstring', widget)
            CONFIG.set_shortcut(hide_doc_action, 'python_hide_docstring')
            hide_doc_action.triggered.connect(self.help_dock.hide)
            widget.editor.addAction(hide_doc_action)
            widget.toolbar.addAction(doc_action)

        self.setCentralWidget(widget)

    def cursor_position_changed(self):
        self.centralWidget().cursor_position_changed()

    def save_state(self):
        try:
            CONFIG['session/scriptwindow'] = self.saveState()
        except AttributeError:
            pass
        else:
            CONFIG.sync()

    def open_help(self):
        open_help('api', self._document.window)

    def show_docstring(self):
        if CONFIG['workarounds/no_jedi']: return
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.EndOfWord)
        row = cursor.blockNumber()
        col = cursor.positionInBlock()
        # QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        if CONFIG['workarounds/blocking_jedi']:
            result = get_docstring(self._document, self.editor.toPlainText(), row, col)
            if type(result) is tuple:
                self.help_dock.show_help(*result)
            else:
                self.help_dock.show_help(result)
        else:
            task = BackgroundTask(lambda: get_docstring(self._document, self.editor.toPlainText(),
                                                        row, col),
                                  self.help_dock.show_help)
            task.start()


class HelpDock(QDockWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.textarea = QTextEdit()
        self.textarea.setReadOnly(True)
        help_font = QFont(EDITOR_FONT)
        set_font(help_font, 'editor/help_font')
        pal = self.textarea.palette()
        pal.setColor(QPalette.ColorRole.Base, QColor(CONFIG['editor/help_background_color']))
        pal.setColor(QPalette.ColorRole.Text, QColor(CONFIG['editor/help_foreground_color']))
        self.textarea.setPalette(pal)
        self.textarea.setFont(help_font)
        font_metrics = self.textarea.fontMetrics()
        self.textarea.setMinimumWidth(86 * font_metrics.horizontalAdvance('a'))
        self.textarea.setMinimumHeight(8 * font_metrics.height())
        self.setWidget(self.textarea)
        self.setObjectName('help')

    def show_help(self, name=None, docstring=None):
        if docstring is not None:
            self.setWindowTitle("Help: " + name)
            self.textarea.setText(docstring)
            self.show()
        # QApplication.restoreOverrideCursor()

    def reconfig(self):
        help_font = self.textarea.font()
        set_font(help_font, 'editor/help_font')
        self.textarea.setFont(help_font)
        font_metrics = self.textarea.fontMetrics()
        self.textarea.setMinimumWidth(86 * font_metrics.horizontalAdvance('a'))
        self.textarea.setMinimumHeight(8 * font_metrics.height())
        self.textarea.setStyleSheet("QTextEdit {{ color: {fg}; background-color: {bg} }}".format(
            fg=(CONFIG['editor/help_foreground_color']),
            bg=(CONFIG['editor/help_background_color'])
        ))
