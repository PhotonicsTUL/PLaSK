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

import math

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from ..qt import QtSignal, qt_exec
from .config import CONFIG, dark_style
from .widgets import EDITOR_FONT, set_icon_size


def update_textedit():
    global CURRENT_LINE_COLOR, SELECTION_COLOR, LINENUMBER_BACKGROUND_COLOR, LINENUMBER_FOREGROUND_COLOR
    CURRENT_LINE_COLOR = QColor(CONFIG['editor/current_line_color'])
    SELECTION_COLOR = QColor(CONFIG['editor/selection_color'])
    LINENUMBER_BACKGROUND_COLOR = QColor(CONFIG['editor/linenumber_background_color'])
    LINENUMBER_FOREGROUND_COLOR = QColor(CONFIG['editor/linenumber_foreground_color'])
    global MATCH_COLOR, REPLACE_COLOR
    MATCH_COLOR = QColor(CONFIG['editor/match_color'])
    REPLACE_COLOR = QColor(CONFIG['editor/replace_color'])
    global SELECT_AFTER_PASTE
    SELECT_AFTER_PASTE = CONFIG['editor/select_after_paste']
update_textedit()


class TextEditor(QPlainTextEdit):
    """Improved editor with line numbers and some other neat stuff"""

    def __init__(self, parent=None, line_numbers=True):
        super().__init__(parent)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(CONFIG['editor/background_color']))
        palette.setColor(QPalette.ColorRole.Text, QColor(CONFIG['editor/foreground_color']))
        self.setFont(EDITOR_FONT)
        self.setPalette(palette)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        if line_numbers:
            self.line_numbers = LineNumberArea(self)
            self.line_numbers.setFont(EDITOR_FONT)
            self.line_numbers.update_width()
            self.blockCountChanged.connect(self.line_numbers.update_width)
            self.updateRequest.connect(self.line_numbers.on_update_request)
        else:
            self.line_numbers = None
        self.cursorPositionChanged.connect(self.update_selections)
        self.selectionChanged.connect(self.update_selections)
        self.selections = []
        self._changed_pos = 0
        self.textChanged.connect(self.on_text_change)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.line_numbers is not None:
            cr = self.contentsRect()
            self.line_numbers.setGeometry(QRect(cr.left(), cr.top(),
                                                       self.line_numbers.get_width(), cr.height()))

    def insertFromMimeData(self, source):
        if source.hasText() and SELECT_AFTER_PASTE:
            cursor = self.textCursor()
            start = min(cursor.position(), cursor.anchor())
            end = start + len(source.text())
            super().insertFromMimeData(source)
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(cursor)
        else:
            super().insertFromMimeData(source)

    def on_text_change(self):
        self._changed_pos = self.textCursor().position()

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        char = event.text()
        if key == Qt.Key.Key_Backspace and modifiers == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            cursor = self.textCursor()
            cursor.setPosition(self._changed_pos)
            self.setTextCursor(cursor)
            event.ignore()
            return
        elif key == Qt.Key.Key_Up and modifiers == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            self.move_line_up()
            event.ignore()
            return
        elif key == Qt.Key.Key_Down and modifiers == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            self.move_line_down()
            event.ignore()
            return
        elif char and char in '([{"\'':
            cursor = self.textCursor()
            if cursor.hasSelection():
                pos, anchor = cursor.position() + 1, cursor.anchor() + 1
                text = char + cursor.selectedText() + {'(': ')', '[': ']', '{': '}', '"': '"', "'": "'"}[char]
                cursor.insertText(text)
                cursor.setPosition(anchor)
                cursor.setPosition(pos, QTextCursor.MoveMode.KeepAnchor)
                self.setTextCursor(cursor)
                event.ignore()
                return

        super().keyPressEvent(event)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.update_selections()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.update_selections()

    def update_selections(self, selections=None):
        """Add our own custom selections"""
        if selections is not None:
            self.selections = selections
        self.setExtraSelections(self.highlight_current_line() + self.get_same_as_selected() + self.selections)

    def move_line_up(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = dst = cursor.selectionEnd()
        if start != end:
            cursor.setPosition(end)
            if cursor.positionInBlock() == 0:
                dst -= 1
        cursor.setPosition(start)
        if not cursor.movePosition(QTextCursor.MoveOperation.PreviousBlock):
            cursor.endEditBlock()
            return
        src = cursor.position()
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        line = cursor.selectedText()
        cursor.setPosition(dst)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        cursor.insertText('\n'+line)
        cursor.setPosition(src)
        cursor.movePosition(QTextCursor.MoveOperation.NextBlock, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.endEditBlock()
        cursor.setPosition(start-len(line)-1)
        if start != end:
            cursor.setPosition(end-len(line)-1, QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cursor)

    def move_line_down(self):
        cursor = self.textCursor()
        cursor.beginEditBlock()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.setPosition(end)
        if start != end and cursor.positionInBlock() == 0:
            cursor.movePosition(QTextCursor.MoveOperation.PreviousCharacter)
        if not cursor.movePosition(QTextCursor.MoveOperation.NextBlock):
            cursor.endEditBlock()
            return
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        line = cursor.selectedText()
        cursor.removeSelectedText()
        cursor.deletePreviousChar()  # remove previous newline
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        cursor.insertText(line+'\n')
        cursor.endEditBlock()
        cursor.setPosition(start+len(line)+1)
        if start != end:
            cursor.setPosition(end+len(line)+1, QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cursor)

    def highlight_current_line(self):
        selection = QTextEdit.ExtraSelection()
        if self.hasFocus():
            selection.format.setBackground(CURRENT_LINE_COLOR)
        selection.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        return [selection]

    def get_same_as_selected(self):
        cursor = self.textCursor()
        if not cursor.hasSelection(): return []
        document = self.document()
        text = cursor.selectedText()
        if not text.strip(): return []
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        selections = []
        while True:
            cursor = document.find(text, cursor,
                                   QTextDocument.FindFlag.FindCaseSensitively | QTextDocument.FindFlag.FindWholeWords)
            if not cursor.isNull():
                selection = QTextEdit.ExtraSelection()
                selection.cursor = cursor
                selection.format.setBackground(SELECTION_COLOR)
                selections.append(selection)
            else:
                break
        return selections

    def reconfig(self):
        self.setFont(EDITOR_FONT)
        self.setStyleSheet("QPlainTextEdit {{ color: {fg}; background-color: {bg} }}".format(
            fg=CONFIG['editor/foreground_color'], bg=CONFIG['editor/background_color']
        ))
        self.line_numbers.setFont(EDITOR_FONT)


class TextEditorWithCB(TextEditor):
    """
        TextEditor which emits some extra callbacks:
        focus_out_cb - when it lost focus
        key_cb - when kay is pressed
    """
    def __init__(self, focus_out_cb=None, key_cb=None, **kwargs):
        super().__init__(**kwargs)
        self.focus_out_cb = focus_out_cb
        self.key_cb = key_cb

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if self.focus_out_cb is not None: self.focus_out_cb()

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if self.key_cb is not None: self.key_cb(event)


class LineNumberArea(QWidget):
    """Line numbers widget

       http://qt4-project.org/doc/qt4-4.8/widgets-codeeditor.html
    """

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self._offset = 0
        self._count_cache = -1, -1

    def get_width(self):
        """Return required width"""
        count = max(1, self.editor.blockCount() + self._offset)
        digits = int(math.log10(count)) + 1
        width = self.editor.fontMetrics().horizontalAdvance('9')
        return 8 + width * digits

    def sizeHint(self):
        QSize(self.get_width(), 0)

    def update_width(self, n=0):
        self.editor.setViewportMargins(self.get_width(), 0, 0, 0)

    def on_update_request(self, rect, dy):
        if dy:
            self.scroll(0, dy)
        elif self._count_cache[0] != self.editor.blockCount() or\
             self._count_cache[1] != self.editor.textCursor().block().lineCount():
            self.update(0, rect.y(), self.width(), rect.height())
            self._count_cache = self.editor.blockCount(), self.editor.textCursor().block().lineCount()
        if rect.contains(self.editor.viewport().rect()):
            self.update_width()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), LINENUMBER_BACKGROUND_COLOR)
        block = self.editor.firstVisibleBlock()
        block_number = block.blockNumber() + 1 + self._offset
        top = self.editor.blockBoundingGeometry(block).translated(self.editor.contentOffset()).top()
        bottom = top + self.editor.blockBoundingRect(block).height()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(LINENUMBER_FOREGROUND_COLOR)
                painter.drawText(0, top, self.width()-3, self.editor.fontMetrics().height(),
                                 Qt.AlignmentFlag.AlignRight, str(block_number))
            block = block.next()
            top = bottom
            bottom = top + self.editor.blockBoundingRect(block).height()
            block_number += 1

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, val):
        if val is not None:
            self._offset = val
            self.update_width()


class LineEditWithHistory(QLineEdit):

    historyChanged = QtSignal()

    def __init__(self, parent=None, flags=None):
        super().__init__(parent, flags)
        self._history = []
        self._position = 1
        self.textEdited.connect(self._update_history)
        self.editingFinished.connect(self.commit_history)

    def _update_history(self):
        text = self.text()
        if self._position == len(self._history):
            if text:
                self._history[-1] = text
            else:
                del self._history[-1]
        elif text:
            self._history.append(text)
            self._position = len(self._history)

    def commit_history(self):
        self._position = len(self._history) + 1

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up and self._position > 1:
            self._position -= 1
            self.setText(self._history[self._position-1])
            self.historyChanged.emit()
        elif event.key() == Qt.Key.Key_Down and self._position < len(self._history):
            self.setText(self._history[self._position])
            self._position += 1
            self.historyChanged.emit()
        super().keyPressEvent(event)


class EditorWidget(QWidget):

    def __init__(self, parent, editor_class=TextEditor, *args, **kwargs):
        super().__init__(parent)

        self.editor = editor_class(self, *args, **kwargs)

        self.toolbar = QToolBar(self)
        self.toolbar.setStyleSheet("QToolBar { border: 0px }")
        set_icon_size(self.toolbar)

        self.add_action('&Undo', 'edit-undo', None, self.editor.undo)
        self.add_action('R&edo', 'edit-redo', None, self.editor.redo)
        self.toolbar.addSeparator()
        self.add_action('&Copy', 'edit-copy', None, self.editor.copy)
        self.add_action('C&ut', 'edit-cut', None, self.editor.cut)
        self.add_action('&Paste', 'edit-paste', None, self.editor.paste)
        self.toolbar.addSeparator()
        self.add_action('&Find...', 'edit-find', 'editor_find', self.show_find)
        self.add_action('&Replace...', 'edit-find-replace', 'editor_replace', self.show_replace)

        self.make_find_replace_widget()

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.editor)
        layout.addWidget(self.message_toolbar)
        layout.addWidget(self.find_toolbar)
        layout.addWidget(self.replace_toolbar)

        layout.setContentsMargins(0, 1, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)

    def make_find_replace_widget(self):
        self.find_toolbar = QToolBar(self)
        self.replace_toolbar = QToolBar(self)
        self.message_toolbar = QToolBar(self)
        self.find_toolbar.setStyleSheet("QToolBar { border: 0px }")
        self.replace_toolbar.setStyleSheet("QToolBar { border: 0px }")
        if dark_style():
            self.message_toolbar.setStyleSheet("QToolBar { border: 1px solid palette(dark);"
                                               "           background-color: #000; color: #ccc; }")
        else:
            self.message_toolbar.setStyleSheet("QToolBar { border: 1px solid palette(dark);"
                                               "           background-color: #ffffcc; color: black; }")
        find_label = QLabel()
        find_label.setText("Search: ")
        find_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        replace_label = QLabel()
        replace_label.setText("Replace: ")
        replace_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label_width = replace_label.fontMetrics().horizontalAdvance(replace_label.text())
        find_label.setFixedWidth(label_width)
        replace_label.setFixedWidth(label_width)
        self.find_edit = LineEditWithHistory()
        self.find_edit.historyChanged.connect(self.find_type)
        self.find_toolbar.addWidget(find_label)
        self.find_toolbar.addWidget(self.find_edit)
        self.replace_edit = LineEditWithHistory()
        self.replace_toolbar.addWidget(replace_label)
        self.replace_toolbar.addWidget(self.replace_edit)

        self.replace_message = QLabel()
        self.message_toolbar.addWidget(self.replace_message)

        self.find_matchcase = QAction('&Match Case', self.find_edit)
        self.find_matchcase.setCheckable(True)
        self.find_matchcase.setChecked(True)
        self.find_wholewords = QAction('&Whole Words', self.find_edit)
        self.find_wholewords.setCheckable(True)
        self.find_regex = QAction('&Regular Expression', self.find_edit)
        self.find_regex.setCheckable(True)
        self.find_regex.triggered.connect(self.trigger_regex)
        self.find_selection = QAction('&Selection Only', self.find_edit)
        self.find_selection.setCheckable(True)
        self.find_edit.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.find_edit.customContextMenuRequested.connect(self._find_context_menu)
        self.find_options = QMenu()
        self.find_options.addAction(self.find_matchcase)
        self.find_options.addAction(self.find_wholewords)
        self.find_options.addAction(self.find_regex)
        self.find_options.addAction(self.find_selection)
        options_button = QPushButton(self)
        options_button.setText("&Options")
        options_button.setMenu(self.find_options)
        self.find_toolbar.addWidget(options_button)

        next_button = QPushButton(self)
        next_button.setText("&Next")
        next_button.pressed.connect(self.find_next)
        prev_button = QPushButton(self)
        prev_button.setText("&Previous")
        prev_button.pressed.connect(self.find_prev)

        self._replace_count = 0

        self.find_toolbar.addWidget(next_button)
        self.find_toolbar.addWidget(prev_button)
        replace_button = QPushButton(self)
        replace_button.setText("Rep&lace one")
        replace_button.pressed.connect(self.replace_next)
        replace_all_button = QPushButton(self)
        replace_all_button.setText("Replace &all")
        replace_all_button.pressed.connect(self.replace_all)
        width = int(replace_button.fontMetrics().horizontalAdvance(replace_button.text()) * 1.2)
        next_button.setFixedWidth(width)
        prev_button.setFixedWidth(width)
        replace_button.setFixedWidth(width)
        replace_all_button.setFixedWidth(width)
        self.replace_toolbar.addWidget(replace_button)
        self.replace_toolbar.addWidget(replace_all_button)
        self.find_toolbar.hide()
        self.replace_toolbar.hide()
        self.message_toolbar.hide()
        hide_action = QAction(self)
        hide_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        hide_action.triggered.connect(self.hide_toolbars)
        self.editor.addAction(hide_action)
        self._add_shortcut('editor_find_next', self.find_next)
        self._add_shortcut('editor_find_prev', self.find_prev)
        self.find_edit.textEdited.connect(self.find_type)
        self.find_edit.returnPressed.connect(self.find_next)
        self.replace_edit.returnPressed.connect(self.replace_next)
        self._replaced_selections = []

    def _find_context_menu(self, pos):
        menu = self.find_edit.createStandardContextMenu()
        menu.addSeparator()
        menu.addAction(self.find_matchcase)
        menu.addAction(self.find_wholewords)
        menu.addAction(self.find_regex)
        menu.addAction(self.find_selection)
        qt_exec(menu, self.find_toolbar.mapToGlobal(pos))

    def _find_flags(self):
        flags = QTextDocument.FindFlag(0)
        if self.find_matchcase.isChecked(): flags |= QTextDocument.FindFlag.FindCaseSensitively
        if self.find_wholewords.isChecked(): flags |= QTextDocument.FindFlag.FindWholeWords
        return flags

    def add_action(self, name, icon, shortcut, slot):
        action = QAction(QIcon.fromTheme(icon), name, self)
        if shortcut is not None:
            CONFIG.set_shortcut(action, shortcut)
        action.triggered.connect(slot)
        self.toolbar.addAction(action)
        return action

    def _add_shortcut(self, shortcut, slot):
        action = QAction(self)
        CONFIG.set_shortcut(action, shortcut)
        action.triggered.connect(slot)
        self.editor.addAction(action)
        return action

    def show_find(self):
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            if u'\u2029' in text:
                self.find_selection.setChecked(True)
            else:
                self.find_selection.setChecked(False)
                self.find_edit.setText(text)
        else:
            self.find_selection.setChecked(False)
        self.find_edit.selectAll()
        self.find_edit.setPalette(self.replace_edit.palette())
        self.find_toolbar.show()
        self.find_edit.setFocus()

    def show_replace(self):
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            text = cursor.selectedText()
            if u'\u2029' in text:
                self.find_selection.setChecked(True)
            else:
                self.find_selection.setChecked(False)
                self.find_edit.setText(text)
        self.find_edit.selectAll()
        self.find_edit.setPalette(self.replace_edit.palette())
        self.find_toolbar.show()
        self.replace_toolbar.show()
        self.find_edit.setFocus()

    def hide_toolbars(self):
        self.find_edit.commit_history()
        self.replace_edit.commit_history()
        self.find_toolbar.hide()
        self.replace_toolbar.hide()
        self.message_toolbar.hide()
        self.clear_matches()
        self.editor.setFocus()

    def _highlight_matches(self):
        cursor = self.editor.textCursor()
        if not cursor.hasSelection(): return []
        document = self.editor.document()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        selections = []
        if self.find_regex.isChecked():
            findtext = QRegularExpression(self.find_edit.text())
        else:
            findtext = self.find_edit.text()
        while True:
            cursor = document.find(findtext, cursor, self._find_flags())
            if not cursor.isNull():
                if not cursor.selection().isEmpty():
                    selection = QTextEdit.ExtraSelection()
                    selection.cursor = cursor
                    selection.format.setBackground(MATCH_COLOR)
                    selections.append(selection)
                else:
                    if not cursor.movePosition(QTextCursor.MoveOperation.NextCharacter):
                        break
            else:
                break
        self.editor.update_selections(selections + self._replaced_selections)

    def clear_matches(self):
        self._replaced_selections = []
        self.editor.update_selections([])

    def _find(self, cont=False, backward=False, rewind=True, theend=None):
        cursor = self.editor.textCursor()
        if cont:
            cursor.setPosition(cursor.selectionStart())
        pal = self.replace_edit.palette()
        if self.find_regex.isChecked():
            self._findtext = QRegularExpression(self.find_edit.text())
        else:
            self._findtext = self.find_edit.text()
        if self._findtext:
            document = self.editor.document()
            findflags = self._find_flags()
            if backward: findflags |= QTextDocument.FindFlag.FindBackward
            found = document.find(self._findtext, cursor, findflags)
            if found.isNull() and rewind:
                cursor.movePosition(QTextCursor.MoveOperation.End if backward else QTextCursor.MoveOperation.Start)
                found = document.find(self._findtext, cursor, findflags)
            if found.isNull():
                pal.setColor(QPalette.ColorRole.Base, QColor("#381111" if dark_style() else "#fdd"))
                self.find_edit.setPalette(pal)
                return False
            elif theend is not None and found.selectionEnd() > theend:
                pal.setColor(QPalette.ColorRole.Base, QColor("#381111" if dark_style() else "#fdd"))
                self.find_edit.setPalette(pal)
                return False
            else:
                self.editor.setTextCursor(found)
                pal.setColor(QPalette.ColorRole.Base, QColor("#232" if dark_style() else "#dfd"))
                self.find_edit.setPalette(pal)
                self._highlight_matches()
                return True
        else:
            self.find_edit.setPalette(pal)
            cursor.setPosition(cursor.position())
            self.editor.setTextCursor(cursor)

    def find_next(self):
        self.find_edit.commit_history()
        self._find()
        #self.editor.setFocus()

    def find_prev(self):
        self.find_edit.commit_history()
        self._find(backward=True)
        #self.editor.setFocus()

    def find_type(self):
        if not self.find_selection.isChecked():
            self._find(cont=True)

    def trigger_regex(self):
        if self.find_toolbar.isVisible():
            self.find_type()

    def _replace_regexp(self, cursor):
        block = cursor.block()
        match = self._findtext.match(block.text(), cursor.selectionStart()-block.position())  # guaranteed to succeed
        text = self.replace_edit.text()
        result = ""
        part = 0
        escape = False
        dollar = False
        for i, c in enumerate(text):
            if dollar and c.isdigit():
                result += match.captured(int(c))
                part = i + 1
                dollar = False
                continue
            dollar = False
            if c == '\\':
                result += text[part:i]
                part = i + 1
                escape = not escape
                if not escape:
                    result += '\\'
            elif c == '$' and not escape:
                result += text[part:i]
                part = i + 1
                dollar = True
            else:
                if escape:
                    result += text[part:i]
                    part = i + 1
                    result += ("\\"+c).encode('raw_unicode_escape').decode('unicode_escape')
                escape = False
        result += text[part:]
        return result

    def replace_next(self):
        self.find_edit.commit_history()
        self.replace_edit.commit_history()
        self.message_toolbar.hide()
        self.clear_matches()
        self._replace_next()

    def _replace_next(self, rewind=True, theend=None):
        if theend is None and self.find_selection.isChecked():
            cursor = self.editor.textCursor()
            if cursor.hasSelection():
                theend = cursor.selectionEnd()
        if rewind:
            self._replaced_selections = []
        if not self._find(cont=True, rewind=rewind, theend=theend):
            return False
        pal = self.editor.palette()
        self.find_edit.setPalette(pal)
        cursor = self.editor.textCursor()
        start = cursor.selectionStart()
        oldlen = self.editor.document().characterCount()
        if isinstance(self._findtext, QRegularExpression):
            cursor.insertText(self._replace_regexp(cursor))
        else:
            cursor.insertText(self.replace_edit.text())
        if theend is not None:
            theend += self.editor.document().characterCount() - oldlen
        end = cursor.position()
        selection = QTextEdit.ExtraSelection()
        selection.cursor = self.editor.textCursor()
        selection.cursor.setPosition(start)
        selection.cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        selection.format.setBackground(REPLACE_COLOR)
        self._replaced_selections.append(selection)
        self._replace_count += 1
        if not self._find(cont=False, rewind=rewind, theend=theend):
            self.editor.update_selections(self._replaced_selections)
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            self.editor.setTextCursor(cursor)
            return False
        # self.editor.setFocus()
        return True

    def replace_all(self):
        self.find_edit.commit_history()
        self.replace_edit.commit_history()
        self.message_toolbar.hide()
        self._replaced_selections = []
        cursor = self.editor.textCursor()
        if self.find_selection.isChecked() and cursor.hasSelection():
            start = cursor.selectionStart()
            end = cursor.selectionEnd()
        else:
            start = end = None
        cursor.beginEditBlock()
        self._replace_count = 0
        try:
            if end is None:
                cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.editor.setTextCursor(cursor)
            doclen = self.editor.document().characterCount()
            while self._replace_next(rewind=False, theend=end):
                if end is not None:
                    newlen = self.editor.document().characterCount()
                    end += newlen - doclen
                    doclen = newlen
            if start is not None:
                cursor.setPosition(start)
                cursor.setPosition(end + self.editor.document().characterCount() - doclen, QTextCursor.MoveMode.KeepAnchor)
                self.editor.setTextCursor(cursor)
        finally:
            cursor.endEditBlock()
            # QToolTip.showText(self.replace_edit.mapToGlobal(QPoint(0, -32)),
            #                         "{} replacement{} made".format(self._replace_count,
            #                                                        's' if self._replace_count != 1 else ''))
            self.replace_message.setText("{} replacement{} made".format(self._replace_count,
                                                                        's' if self._replace_count != 1 else ''))
            self.message_toolbar.show()
