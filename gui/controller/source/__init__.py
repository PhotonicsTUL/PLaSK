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

from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from .. import Controller
from ...utils.config import CONFIG, parse_highlight
from ...utils.qsignals import BlockQtSignals
from ...utils.texteditor import TextEditor
from ...utils.widgets import EDITOR_FONT
from ...external.highlighter import SyntaxHighlighter, load_syntax
from ...external.highlighter.xml import syntax
from .indenter import indent, unindent, autoindent


def update_xml_scheme():
    global SCHEME
    SCHEME = {
        'syntax_comment': parse_highlight(CONFIG['syntax/xml_comment']),
        'syntax_tag': parse_highlight(CONFIG['syntax/xml_tag']),
        'syntax_attr': parse_highlight(CONFIG['syntax/xml_attr']),
        'syntax_value': parse_highlight(CONFIG['syntax/xml_value']),
        'syntax_text': parse_highlight(CONFIG['syntax/xml_text']),
    }
update_xml_scheme()

MATCH_COLOR = QColor(CONFIG['editor/match_color'])
REPLACE_COLOR = QColor(CONFIG['editor/replace_color'])


class XMLEditor(TextEditor):

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key_Tab, Qt.Key_Backtab, Qt.Key_Backspace):
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
            mode = QTextCursor.KeepAnchor if modifiers & Qt.ShiftModifier else QTextCursor.MoveAnchor
            if txt[:col].strip():
                cursor.movePosition(QTextCursor.StartOfBlock, mode)
                while self.document().characterAt(cursor.position()) in [' ', '\t']:
                    cursor.movePosition(QTextCursor.Right, mode)
                self.setTextCursor(cursor)
                return

        super(XMLEditor, self).keyPressEvent(event)

        if key in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Greater):
            autoindent(self)


class SourceWidget(QWidget):

    def __init__(self, parent, editor_class=TextEditor, *args, **kwargs):
        super(SourceWidget, self).__init__(parent)

        self.editor = editor_class(self, *args, **kwargs)
        self.editor.setFont(EDITOR_FONT)

        self.toolbar = QToolBar(self)
        self.toolbar.setStyleSheet("QToolBar { border: 0px }")

        self.add_action('&Undo', 'edit-undo', None, self.editor.undo)
        self.add_action('R&edo', 'edit-redo', None, self.editor.redo)
        self.toolbar.addSeparator()
        self.add_action('&Copy', 'edit-copy', None, self.editor.copy)
        self.add_action('C&ut', 'edit-cut', None, self.editor.cut)
        self.add_action('&Paste', 'edit-paste', None, self.editor.paste)
        self.toolbar.addSeparator()
        self.add_action('&Find...', 'edit-find', QKeySequence.Find, self.show_find)
        self.add_action('&Replace...', 'edit-find-replace', QKeySequence.Replace, self.show_replace)

        self.make_find_replace_widget()

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.editor)
        layout.addWidget(self.find_toolbar)
        layout.addWidget(self.replace_toolbar)

        layout.setContentsMargins(0, 1, 0, 0)
        layout.setSpacing(0)

        self.setLayout(layout)

    def make_find_replace_widget(self):
        self.find_toolbar = QToolBar(self)
        self.replace_toolbar = QToolBar(self)
        self.find_toolbar.setStyleSheet("QToolBar { border: 0px }")
        self.replace_toolbar.setStyleSheet("QToolBar { border: 0px }")
        find_label = QLabel()
        find_label.setText("Search: ")
        find_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        replace_label = QLabel()
        replace_label.setText("Replace: ")
        replace_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label_width = replace_label.fontMetrics().width(replace_label.text())
        find_label.setFixedWidth(label_width)
        replace_label.setFixedWidth(label_width)
        self.find_edit = QLineEdit()
        self.find_toolbar.addWidget(find_label)
        self.find_toolbar.addWidget(self.find_edit)
        self.replace_edit = QLineEdit()
        self.replace_toolbar.addWidget(replace_label)
        self.replace_toolbar.addWidget(self.replace_edit)

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
        self.find_edit.setContextMenuPolicy(Qt.CustomContextMenu)
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

        self.find_toolbar.addWidget(next_button)
        self.find_toolbar.addWidget(prev_button)
        replace_button = QPushButton(self)
        replace_button.setText("Rep&lace one")
        replace_button.pressed.connect(self.replace_next)
        replace_all_button = QPushButton(self)
        replace_all_button.setText("Replace &all")
        replace_all_button.pressed.connect(self.replace_all)
        width = int(replace_button.fontMetrics().width(replace_button.text()) * 1.2)
        next_button.setFixedWidth(width)
        prev_button.setFixedWidth(width)
        replace_button.setFixedWidth(width)
        replace_all_button.setFixedWidth(width)
        self.replace_toolbar.addWidget(replace_button)
        self.replace_toolbar.addWidget(replace_all_button)
        self.find_toolbar.hide()
        self.replace_toolbar.hide()
        self._add_shortcut(QKeySequence(Qt.Key_Escape), self.hide_toolbars)
        self._add_shortcut(QKeySequence.FindNext, self.find_next, alt=QKeySequence(Qt.Key_F3))
        self._add_shortcut(QKeySequence.FindPrevious, self.find_prev, alt=Qt.SHIFT+Qt.Key_F3)
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
        menu.exec_(self.find_toolbar.mapToGlobal(pos))

    def _find_flags(self):
        flags = QTextDocument.FindFlags()
        if self.find_matchcase.isChecked(): flags |= QTextDocument.FindCaseSensitively
        if self.find_wholewords.isChecked(): flags |= QTextDocument.FindWholeWords
        return flags

    def add_action(self, name, icon, shortcut, slot):
        action = QAction(QIcon.fromTheme(icon), name, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        self.toolbar.addAction(action)
        return action

    def _add_shortcut(self, shortcut, slot, alt=None):
        action = QAction(self)
        action.setShortcut(shortcut)
        action.triggered.connect(slot)
        if alt is not None and action.shortcut() != alt:
            self._add_shortcut(alt, slot)
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
        self.find_edit.setPalette(self.editor.palette())
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
        self.find_edit.setPalette(self.editor.palette())
        self.find_toolbar.show()
        self.replace_toolbar.show()
        self.find_edit.setFocus()

    def hide_toolbars(self):
        self.find_toolbar.hide()
        self.replace_toolbar.hide()
        self.clear_matches()
        self.editor.setFocus()

    def _highlight_matches(self):
        cursor = self.editor.textCursor()
        if not cursor.hasSelection(): return []
        document = self.editor.document()
        cursor.movePosition(QTextCursor.Start)
        selections = []
        if self.find_regex.isChecked():
            findtext = QRegExp(self.find_edit.text())
        else:
            findtext = self.find_edit.text()
        while True:
            cursor = document.find(findtext, cursor, self._find_flags())
            if not cursor.isNull():
                selection = QTextEdit.ExtraSelection()
                selection.cursor = cursor
                selection.format.setBackground(MATCH_COLOR)
                selections.append(selection)
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
        pal = self.editor.palette()
        if self.find_regex.isChecked():
            self._findtext = QRegExp(self.find_edit.text())
        else:
            self._findtext = self.find_edit.text()
        if self._findtext:
            document = self.editor.document()
            findflags = self._find_flags()
            if backward: findflags |= QTextDocument.FindBackward
            found = document.find(self._findtext, cursor, findflags)
            if found.isNull() and rewind:
                cursor.movePosition(QTextCursor.End if backward else QTextCursor.Start)
                found = document.find(self._findtext, cursor, findflags)
            if found.isNull():
                pal.setColor(QPalette.Base, QColor("#fdd"))
                self.find_edit.setPalette(pal)
                return False
            elif theend is not None and found.selectionEnd() > theend:
                pal.setColor(QPalette.Base, QColor("#fdd"))
                self.find_edit.setPalette(pal)
                return False
            else:
                self.editor.setTextCursor(found)
                pal.setColor(QPalette.Base, QColor("#dfd"))
                self.find_edit.setPalette(pal)
                self._highlight_matches()
                return True
        else:
            self.find_edit.setPalette(pal)
            cursor.setPosition(cursor.position())
            self.editor.setTextCursor(cursor)

    def find_next(self):
        self._find()
        #self.editor.setFocus()

    def find_prev(self):
        self._find(backward=True)
        #self.editor.setFocus()

    def find_type(self):
        if not self.find_selection.isChecked():
            self._find(cont=True)

    def trigger_regex(self):
        if self.find_toolbar.isVisible():
            self. find_type()

    def _replace_regexp(self, cursor):
        block = cursor.block()
        self._findtext.indexIn(block.text(), cursor.selectionStart()-block.position())  # guaranteed to succeed
        text = self.replace_edit.text()
        result = ""
        s = 0
        ignore = False
        for i, c in enumerate(text):
            if text[i] in '0123456789\\' and i != 0 and text[i-1] == '\\' and not ignore:
                result += text[s:i-1]
                s = i + 1
                if text[i] == '\\':
                    result += '\\'
                    ignore = True
                else:
                    g = int(text[i])
                    result += self._findtext.cap(g)
            elif ignore:
                ignore = False
        result += text[s:]
        return result

    def replace_next(self, rewind=True, theend=None):
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
        if isinstance(self._findtext, QRegExp):
            cursor.insertText(self._replace_regexp(cursor))
        else:
            cursor.insertText(self.replace_edit.text())
        if theend is not None:
            theend += self.editor.document().characterCount() - oldlen
        end = cursor.position()
        selection = QTextEdit.ExtraSelection()
        selection.cursor = self.editor.textCursor()
        selection.cursor.setPosition(start)
        selection.cursor.setPosition(end, QTextCursor.KeepAnchor)
        selection.format.setBackground(REPLACE_COLOR)
        self._replaced_selections.append(selection)
        self._replace_count += 1
        if not self._find(cont=False, rewind=rewind, theend=theend):
            self.editor.update_selections(self._replaced_selections)
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            self.editor.setTextCursor(cursor)
            return False
        # self.editor.setFocus()
        return True

    def replace_all(self):
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
                cursor.movePosition(QTextCursor.Start)
            self.editor.setTextCursor(cursor)
            doclen = self.editor.document().characterCount()
            while self.replace_next(rewind=False, theend=end):
                if end is not None:
                    newlen = self.editor.document().characterCount()
                    end += newlen - doclen
                    doclen = newlen
            if start is not None:
                cursor.setPosition(start)
                cursor.setPosition(end + self.editor.document().characterCount() - doclen, QTextCursor.KeepAnchor)
                self.editor.setTextCursor(cursor)
        finally:
            cursor.endEditBlock()
            QToolTip.showText(self.replace_edit.mapToGlobal(QPoint(0, -32)),
                                    "{} replacement{} made".format(self._replace_count,
                                                                   's' if self._replace_count != 1 else ''))


class SourceEditController(Controller):

    def __init__(self, document=None, model=None, line_numbers=True):
        Controller.__init__(self, document, model)
        self.line_numbers = line_numbers
        self.fresh = False
        self.visible = False
        self.source_widget = None
        self.document.window.config_changed.connect(self.reconfig)
        self.highlighter = None

    def create_source_widget(self, parent):
        source = SourceWidget(parent, XMLEditor, line_numbers=self.line_numbers)
        self.highlighter = SyntaxHighlighter(source.editor.document(),
                                             *load_syntax(syntax, SCHEME),
                                             default_font=EDITOR_FONT)
        source.editor.setReadOnly(self.model.is_read_only())
        return source

    def get_source_widget(self):
        if self.source_widget is None:
            self.source_widget = self.create_source_widget(self.document.window)
        return self.source_widget

    # GUI editor, by default use source editor
    def get_widget(self):
        return self.get_source_widget()

    def reconfig(self):
        editor = self.source_widget.editor
        editor.setFont(EDITOR_FONT)
        if editor.line_numbers is not None:
            editor.line_numbers.setFont(EDITOR_FONT)
        if self.highlighter is not None:
            with BlockQtSignals(editor):
                update_xml_scheme()
                self.highlighter = SyntaxHighlighter(editor.document(),
                                                     *load_syntax(syntax, SCHEME),
                                                     default_font=EDITOR_FONT)

    def refresh_editor(self, *args, **kwargs):
        if self.visible:
            editor = self.get_source_widget().editor
            text = self.model.get_text()
            if text and text[-1] == '\n':
                text = text[:-1]
            editor.setPlainText(text)
            self.fresh = True
        else:
            self.fresh = False

    def save_data_in_model(self):
        if not self.get_source_widget().editor.isReadOnly() and \
                self.get_source_widget().editor.document().isModified():
            try: self.model.changed -= self.refresh_editor
            except AttributeError: pass
            try:
                self.model.set_text(self.get_source_widget().editor.toPlainText() + '\n')
            finally:
                try: self.model.changed += self.refresh_editor
                except AttributeError: pass
            self.get_source_widget().editor.document().setModified(False)

    def _modification_changed(self, changed):
        if changed:
            self.document.set_changed()

    def on_edit_enter(self):
        self.visible = True
        if not self.fresh: self.refresh_editor()
        try: self.source_widget.editor.line_numbers.offset = self.model.line_in_file
        except AttributeError: pass
        try: self.model.changed += self.refresh_editor
        except AttributeError: pass
        #self._clean_state = self.model.undo_stack.isClean()
        self.source_widget.editor.modificationChanged.connect(self._modification_changed)

    # When the editor is turned off, the model should be updated
    def on_edit_exit(self):
        if not self.try_save_data_in_model():
            return False
        self.source_widget.editor.modificationChanged.disconnect(self._modification_changed)
        #if hasattr(self.model, 'changed'): self.model.changed -= self.refresh_editor
        self.visible = False
        return True
