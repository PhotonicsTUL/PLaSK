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

import collections

from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
from ..qt import QtSignal, qt_exec
try:
    from ..qt.QtGui import QStyleOptionViewItemV4 as QStyleOptionViewItem
except ImportError:
    from ..qt.QtWidgets import QStyleOptionViewItem
from .qsignals import BlockQtSignals

from ..utils.config import CONFIG, set_font

EDITOR_FONT = QFont()
EDITOR_FONT.setBold(False)
set_font(EDITOR_FONT, 'editor/font')


def set_icon_size(widget):
    setting = CONFIG['main_window/icons_size']
    if setting == 'default': return
    try:
        size = {'small': 16, 'normal': 22, 'large': 32, 'huge': 48, 'enormous': 64}[setting]
    except KeyError:
        pass
    else:
        widget.setIconSize(QSize(size, size))



def table_edit_shortcut(table, col, key):
    table.setEditTriggers(QAbstractItemView.EditTrigger.SelectedClicked | QAbstractItemView.EditTrigger.DoubleClicked)
    def operation():
        selected = table.selectedIndexes()
        if selected:
            table.edit((table.model()).index(selected[0].row(), col))
    shortcut = QShortcut(key, table)
    shortcut.activated.connect(operation)
    shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)


def table_last_col_fill(table, cols_count, col_size=0):
    if isinstance(col_size, collections.Sequence):
        for c in range(0, cols_count-1): table.setColumnWidth(c, col_size[c])
    else:
        for c in range(0, cols_count-1): table.setColumnWidth(c, col_size)
    #table.horizontalHeader().setResizeMode(cols_count-1, QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setStretchLastSection(True)


def create_undo_actions(toolbar, model, widget):
    undo = model.create_undo_action(widget)
    redo = model.create_redo_action(widget)
    toolbar.addAction(undo)
    toolbar.addAction(redo)

    class SetupMenu:
        def __init__(self, action, redo):
            self.button = toolbar.widgetForAction(action)
            self.button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.button.customContextMenuRequested.connect(self)
            self.redo = redo
            self.prefix = ('Undo ', 'Redo ')[redo]
        def __call__(self, pos):
            try:
                undo_stack = model.undo_stack
            except AttributeError:
                return
            if undo_stack is None or undo_stack.count() == 0:
                return
            menu = QMenu()
            items = range(undo_stack.index(), undo_stack.count()) if self.redo else \
                    range(undo_stack.index()-1, -1, -1)
            if len(items) == 0:
                return
            for n,i in enumerate(items):
                if n < 9: nr = " &{}  ".format(n+1)
                else: nr = "{}  ".format(n+1)
                menu.addAction(nr+self.prefix+undo_stack.text(i), lambda i=i: undo_stack.setIndex(i+self.redo))
            qt_exec(menu, self.button.mapToGlobal(pos))

    toolbar._undo_menu = SetupMenu(undo, 0)
    toolbar._redo_menu = SetupMenu(redo, 1)


class HTMLDelegate(QStyledItemDelegate):

    def __init__(self, parent=None, compact=False):
        super().__init__(parent)
        self.compact = compact

    def paint(self, painter, option, index):
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        style = QApplication.style() if options.widget is None else options.widget.style()

        doc = QTextDocument()
        if self.compact:
            text_option = doc.defaultTextOption()
            text_option.setWrapMode(QTextOption.WrapMode.NoWrap)
            doc.setDefaultTextOption(text_option)
        doc.documentLayout().setPaintDevice(self.parent())
        doc.setHtml(options.text)
        doc.setTextWidth(max(300, options.rect.width()))    #TODO 300 -> member

        options.text = ""
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, options, painter);

        ctx = QAbstractTextDocumentLayout.PaintContext()

        # Highlighting text if item is selected
        #if optionV4.state & QStyle.StateFlag.State_Selected:
            #ctx.palette.setColor(QPalette.ColorRole.Text,
            #                     optionV4.palette.color(QPalette.ColorRole.Active, QPalette.ColorRole.HighlightedText));

        textrect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, options, None)
        painter.save()
        topleft = textrect.topLeft()
        topleft.setY(topleft.y() + max(int((textrect.height() - doc.size().height()) / 2), 0))
        painter.translate(topleft)
        painter.setClipRect(textrect.translated(-topleft))
        doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)

        doc = QTextDocument()
        doc.documentLayout().setPaintDevice(self.parent())
        if self.compact:
            height = doc.size().height()
        doc.setHtml(options.text)
        doc.setTextWidth(max(300, options.rect.width()))
        if not self.compact:
            height = doc.size().height()
        return QSize(doc.idealWidth(), height)


class ComboBoxDelegate(QItemDelegate):

    def __init__(self, list, parent, editable=True):
        QItemDelegate.__init__(self, parent)
        self.list = list
        self.editable = editable

    def createEditor(self, parent, option, index):
        combo = ComboBox(parent)
        combo.setEditable(self.editable)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        try:
            self._list = list(self.list())
        except TypeError:
            self._list = self.list
        combo.addItems(self._list)
        self._first_enter = True
        return combo

    def setEditorData(self, combo, index):
        try:
            combo.setCurrentIndex(self._list.index(index.data()))
        except (IndexError, ValueError):
            pass
        if self.editable:
            combo.setEditText(index.data())
        else:
            combo.showPopup()

    # def sizeHint(self, option, index):
    #     hint = super().sizeHint(option, index)
    #     style = self.parent().style()
    #     opt = QStyleOptionComboBox()
    #     opt.initFrom(self.parent())
    #     rc = style.subControlRect(QStyle.ComplexControl.CC_ComboBox, opt, QStyle.SC_ComboBoxArrow, self.parent())
    #     try:
    #         lst = list(self.list())
    #     except TypeError:
    #         lst = self.list
    #     fm = option.fontMetrics
    #     width = max(fm.width(l) for l in lst)
    #     hint.setWidth(width + rc.width() + 4)
    #     print hint.width()

    def eventFilter(self, combo, event):
        if not self.editable and event.type() == QEvent.Type.Enter and self._first_enter:
            combo.showPopup()
            self._first_enter = False
            return True
        else:
            return super().eventFilter(combo, event)


class CheckBoxDelegate(QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        """
        Important, otherwise an editor is created if the user clicks in this cell.
        """
        return None

    def paint(self, painter, option, index):
        """
        Paint a checkbox without the label.
        """
        if not index.model().data(index, Qt.ItemDataRole.UserRole):
            return

        checked = bool(index.model().data(index, Qt.ItemDataRole.DisplayRole))
        check_box_style_option = QStyleOptionButton()

        if index.flags() & Qt.ItemFlag.ItemIsEditable:
            check_box_style_option.state |= QStyle.StateFlag.State_Enabled
        else:
            check_box_style_option.state |= QStyle.StateFlag.State_ReadOnly

        if checked:
            check_box_style_option.state |= QStyle.StateFlag.State_On
        else:
            check_box_style_option.state |= QStyle.StateFlag.State_Off

        check_box_style_option.rect = self.get_check_box_rect(option)
        if not (index.model().flags(index) & Qt.ItemFlag.ItemIsEditable):
            check_box_style_option.state |= QStyle.StateFlag.State_ReadOnly

        QApplication.style().drawControl(QStyle.ControlElement.CE_CheckBox, check_box_style_option, painter)

    def editorEvent(self, event, model, option, index):
        """
        Change the data in the model and the state of the checkbox
        if the user presses the left mousebutton or presses
        Key_Space or Key_Select and this cell is editable. Otherwise do nothing.
        """
        if not (index.flags() & Qt.ItemFlag.ItemIsEditable):
            return False

        # Do not change the checkbox-state
        if event.type() == QEvent.Type.MouseButtonRelease or event.type() == QEvent.Type.MouseButtonDblClick:
            if event.button() != Qt.MouseButton.LeftButton or not self.get_check_box_rect(option).contains(event.pos()):
                return False
            if event.type() == QEvent.Type.MouseButtonDblClick:
                return True
        elif event.type() == QEvent.Type.KeyPress:
            if event.key() != Qt.Key.Key_Space and event.key() != Qt.Key.Key_Select:
                return False
        else:
            return False

        # Change the checkbox-state
        self.setModelData(None, model, index)
        return True

    def setModelData(self, editor, model, index):
        """
        The user wanted to change the old state in the opposite.
        """
        new_value = not bool(index.model().data(index, Qt.ItemDataRole.DisplayRole))
        model.setData(index, new_value, Qt.ItemDataRole.EditRole)

    @staticmethod
    def get_check_box_rect(option):
        check_box_style_option = QStyleOptionButton()
        check_box_rect = QApplication.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator,
                                                                   check_box_style_option, None)
        check_box_point = QPoint (option.rect.x() +
                                         option.rect.width() / 2 -
                                         check_box_rect.width() / 2,
                                         option.rect.y() +
                                         option.rect.height() / 2 -
                                         check_box_rect.height() / 2)
        return QRect(check_box_point, check_box_rect.size())


class VerticalScrollArea(QScrollArea):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setWidgetResizable(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        widget = self.widget()
        if widget is not None:
            widget.setFixedWidth(event.size().width())

    def eventFilter(self, obj, event):
        if obj and obj == self.widget() and event.type() == QEvent.Type.Resize:
            self.setMinimumWidth(obj.minimumSizeHint().width() + self.verticalScrollBar().width())
        return super().eventFilter(obj, event)


class ComboBox(QComboBox):

    def wheelEvent(self, evt):
        evt.ignore()


class EditComboBox(ComboBox):

    editingFinished = QtSignal()

    def focusOutEvent(self, event):
        if not self.signalsBlocked(): self.editingFinished.emit()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return) and not self.signalsBlocked():
            self.editingFinished.emit()

class InfoListView(QListView):

    def __init__(self, model, parent):
        super().__init__(parent)
        self.model = model
        self.setMouseTracking(True)
        self._orig_height = None

    def enterEvent(self, event):
        super().enterEvent(event)
        rows = self.model.rowCount()
        if rows > 0:
            self._orig_height = self.height()
            self.setFixedHeight(self.sizeHintForRow(0) * rows)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self._orig_height is not None:
            self.setFixedHeight(self._orig_height)
            self._orig_height = None


class TextEdit(QTextEdit):
    editingFinished = QtSignal()

    def focusOutEvent(self, event):
        if not self.signalsBlocked(): self.editingFinished.emit()
        super().focusOutEvent(event)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return) and event.modifiers() == Qt.Modifier.CTRL and not self.signalsBlocked():
            self.editingFinished.emit()


class TextEditWithCB(QPlainTextEdit):
    """
        QPlainTextEdit which emits some extra callbacks:
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


from ..controller.defines import get_defines_completer


class MultiLineEdit(QWidget):
    """
    Widget showing multiple lines
    """

    class LineEdit(QLineEdit):
        def focusInEvent(self, event):
            super(MultiLineEdit.LineEdit, self).focusInEvent(event)
            self.deselect()

    class Delegate(QStyledItemDelegate):
        def __init__(self, parent=None, defines=None):
            super(MultiLineEdit.Delegate, self).__init__(parent)
            self.defines = defines
        def createEditor(self, parent, option, index):
            editor = MultiLineEdit.LineEdit(parent)
            editor.setStyleSheet("border: 1px solid #888")
            if self.defines is not None:
                editor.setCompleter(get_defines_completer(self.defines, editor))
            return editor
        def initStyleOption(self, option, index):
            super(MultiLineEdit.Delegate, self).initStyleOption(option, index)
            option.state &= ~QStyle.StateFlag.State_MouseOver
            if option.state & QStyle.StateFlag.State_Selected:
                option.state |= QStyle.StateFlag.State_MouseOver
            if not self.parent().hasFocus():
                option.state &= ~QStyle.StateFlag.State_Selected

    def __init__(self, movable=False, change_cb=None, document=None, compact=True):
        super().__init__()
        self.change_cb = change_cb
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self.selected)
        layout.addWidget(self.list_widget)
        buttons = QGridLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(1)
        add = QToolButton()
        add.setIcon(QIcon.fromTheme('list-add'))
        add.pressed.connect(self.add_item)
        buttons.addWidget(add, 0, 0)
        act = QAction(self.list_widget)
        act.setShortcut(QKeySequence(Qt.Key.Key_Plus))
        act.triggered.connect(self.add_item)
        act.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self.list_widget.addAction(act)
        self.remove = QToolButton()
        self.remove.setIcon(QIcon.fromTheme('list-remove'))
        self.remove.pressed.connect(self.remove_item)
        self._compact = compact
        if document is not None:
            defines = document.defines.model
        else:
            defines = None
        if compact:
            buttons.addWidget(self.remove, 1, 0)
            add.setStyleSheet('border: none;')
            self.remove.setStyleSheet('border: none;')
            delegate = MultiLineEdit.Delegate(self.list_widget, defines)
            self.list_widget.setItemDelegate(delegate)
            self.list_widget.setEditTriggers(QAbstractItemView.EditTrigger.CurrentChanged |
                                             QAbstractItemView.EditTrigger.SelectedClicked |
                                             QAbstractItemView.EditTrigger.DoubleClicked |
                                             QAbstractItemView.EditTrigger.EditKeyPressed)
            self.list_widget.setFixedHeight(60)
            self.edit = None
        else:
            self.edit = QToolButton()
            self.edit.setIcon(QIcon.fromTheme('document-edit'))
            self.edit.pressed.connect(self.edit_item)
            buttons.addWidget(self.edit, 1, 0)
            buttons.addWidget(self.remove, 2, 0)
            self.list_widget.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked |
                                             QAbstractItemView.EditTrigger.EditKeyPressed)
        act = QAction(self.list_widget)
        act.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        act.triggered.connect(self.remove_item)
        act.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self.list_widget.addAction(act)
        self._movable = movable
        if movable:
            self.up = QToolButton()
            self.up.setIcon(QIcon.fromTheme('go-up'))
            self.up.pressed.connect(self.move_up)
            act = QAction(self.list_widget)
            act.setShortcut(Qt.Modifier.CTRL + Qt.Modifier.SHIFT + Qt.Key.Key_Up)
            act.triggered.connect(self.move_up)
            act.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
            self.list_widget.addAction(act)
            self.down = QToolButton()
            self.down.setIcon(QIcon.fromTheme('go-down'))
            self.down.pressed.connect(self.move_down)
            act = QAction(self.list_widget)
            act.setShortcut(Qt.Modifier.CTRL + Qt.Modifier.SHIFT + Qt.Key.Key_Down)
            act.triggered.connect(self.move_down)
            act.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
            self.list_widget.addAction(act)
            if compact:
                self.up.setStyleSheet('border: none;')
                self.down.setStyleSheet('border: none;')
                buttons.addWidget(self.up, 0, 1)
                buttons.addWidget(self.down, 1, 1)
            else:
                buttons.addWidget(self.up, 3, 0)
                buttons.addWidget(self.down, 4, 0)
        else:
            self.up = None
            self.down = None
        layout.addLayout(buttons)
        self.selected()
        # act = QAction(self.list_widget)
        # act.setShortcut(QKeySequence(Qt.Key.Key_Return))
        # act.triggered.connect(self.edit_item)
        # act.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        # self.list_widget.addAction(act)
        self.list_widget.itemSelectionChanged.connect(self.selected)
        self.list_widget.itemChanged.connect(self.item_changed)

    def disable_actions(self):
        self.remove.setEnabled(False)
        if self.edit is not None: self.edit.setEnabled(False)
        if self.up is not None: self.up.setEnabled(False)
        if self.down is not None: self.down.setEnabled(False)

    def selected(self):
        row = self.list_widget.currentRow()
        if row != -1:
            self.remove.setEnabled(True)
            if self.edit is not None: self.edit.setEnabled(True)
            if self.up is not None: self.up.setEnabled(row > 0)
            if self.down is not None: self.down.setEnabled(row < self.list_widget.count()-1)
        else:
            self.disable_actions()

    def item_changed(self, item):
        if item is not None:
            self.list_widget.setCurrentItem(item)
        if self.change_cb is not None:
            self.change_cb()
        self.selected()

    def add_item(self):
        with BlockQtSignals(self.list_widget):
            if self._compact:
                item = QListWidgetItem()
            else:
                item = QListWidgetItem('[enter value]')
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            current = self.list_widget.currentRow()
            if self._movable and current != -1:
                self.list_widget.insertItem(current+1, item)
            else:
                self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(None)
        self.list_widget.scrollToItem(item)
        if self._compact:
            self.remove.setEnabled(True)
        self.list_widget.editItem(item)

    def edit_item(self):
        item = self.list_widget.currentItem()
        if item:
            self.list_widget.editItem(item)

    def remove_item(self):
        row = self.list_widget.currentRow()
        if row != -1:
            self.list_widget.takeItem(row)
            self.item_changed(None)

    def move_up(self):
        row = self.list_widget.currentRow()
        if row > 0:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row-1, item)
            self.item_changed(item)

    def move_down(self):
        row = self.list_widget.currentRow()
        if row != -1 and row < self.list_widget.count()-1:
            item = self.list_widget.takeItem(row)
            self.list_widget.insertItem(row+1, item)
            self.item_changed(item)

    def set_values(self, values):
        values = list(values)
        n = len(values)
        with BlockQtSignals(self.list_widget):
            if self.list_widget.count() != n:
                self.list_widget.clear()
                self.list_widget.addItems(values)
                for i in range(n):
                    item = self.list_widget.item(i)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            else:
                for i in range(n):
                    item = self.list_widget.item(i)
                    item.setText(values[i])

    def get_values(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]


def fire_edit_end(widget=None):
    """
    Try to call event which cause updating model by widget which is focused (or widget given as parameter).
    :param QWidget widget: QApplication.focusWidget() will be used by default
    """
    if widget is None: widget = QApplication.focusWidget()
    try: widget.editingFinished.emit()
    except: pass
    try: widget.focus_out_cb()
    except: pass


class LineEditWithClear(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        clear_button = QToolButton(self)
        clear_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_button.setIcon(QIcon(QIcon.fromTheme("edit-clear")))
        clear_button.setStyleSheet("background: transparent; border: none;")
        clear_button.clicked.connect(self.clear)
        layout = QHBoxLayout(self)
        layout.addWidget(clear_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.setSpacing(0)
        layout.setContentsMargins(5, 5, 5, 5)
