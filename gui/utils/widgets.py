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

from ..qt.QtCore import Qt
from ..qt import QtCore, QtGui, QtSignal

from .qsignals import BlockQtSignals

from ..utils.config import CONFIG


EDITOR_FONT = QtGui.QFont()
EDITOR_FONT.setBold(False)
EDITOR_FONT.fromString(','.join(CONFIG['editor/font']))


def table_edit_shortcut(table, col, key):
    table.setEditTriggers(QtGui.QAbstractItemView.SelectedClicked | QtGui.QAbstractItemView.DoubleClicked)
    def operation():
        selected = table.selectedIndexes()
        if selected:
            table.edit((table.model()).index(selected[0].row(), col))
    shortcut = QtGui.QShortcut(key, table)
    shortcut.activated.connect(operation)
    shortcut.setContext(Qt.WidgetShortcut)


def table_last_col_fill(table, cols_count, col_size=0):
    if isinstance(col_size, collections.Sequence):
        for c in range(0, cols_count-1): table.setColumnWidth(c, col_size[c])
    else:
        for c in range(0, cols_count-1): table.setColumnWidth(c, col_size)
    #table.horizontalHeader().setResizeMode(cols_count-1, QtGui.QHeaderView.Stretch)
    table.horizontalHeader().setStretchLastSection(True)


def create_undo_actions(toolbar, model, widget):
    undo = model.create_undo_action(widget)
    redo = model.create_redo_action(widget)
    toolbar.addAction(undo)
    toolbar.addAction(redo)

    class SetupMenu(object):
        def __init__(self, action, redo):
            self.button = toolbar.widgetForAction(action)
            self.button.setContextMenuPolicy(Qt.CustomContextMenu)
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
            menu = QtGui.QMenu()
            items = range(undo_stack.index(), undo_stack.count()) if self.redo else \
                    range(undo_stack.index()-1, -1, -1)
            if len(items) == 0:
                return
            for n,i in enumerate(items):
                if n < 9: nr = " &{}  ".format(n+1)
                else: nr = "{}  ".format(n+1)
                menu.addAction(nr+self.prefix+undo_stack.text(i), lambda i=i: undo_stack.setIndex(i+self.redo))
            menu.exec_(self.button.mapToGlobal(pos))

    toolbar._undo_menu = SetupMenu(undo, 0)
    toolbar._redo_menu = SetupMenu(redo, 1)


class HTMLDelegate(QtGui.QStyledItemDelegate):

    def paint(self, painter, option, index):
        options = QtGui.QStyleOptionViewItemV4(option)
        self.initStyleOption(options, index)

        style = QtGui.QApplication.style() if options.widget is None else options.widget.style()

        doc = QtGui.QTextDocument()
        doc.documentLayout().setPaintDevice(self.parent())
        doc.setHtml(options.text)
        doc.setTextWidth(max(300, options.rect.width()))    #TODO 300 -> member

        options.text = ""
        style.drawControl(QtGui.QStyle.CE_ItemViewItem, options, painter);

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        # Highlighting text if item is selected
        #if (optionV4.state & QStyle::State_Selected)
            #ctx.palette.setColor(QPalette::Text, optionV4.palette.color(QPalette::Active, QPalette::HighlightedText));

        textrect = style.subElementRect(QtGui.QStyle.SE_ItemViewItemText, options, None)
        painter.save()
        topleft = textrect.topLeft()
        topleft.setY(topleft.y() + max(int((textrect.height() - doc.size().height()) / 2), 0))
        painter.translate(topleft)
        painter.setClipRect(textrect.translated(-topleft))
        doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        options = QtGui.QStyleOptionViewItemV4(option)
        self.initStyleOption(options, index)

        doc = QtGui.QTextDocument()
        doc.documentLayout().setPaintDevice(self.parent())
        doc.setHtml(options.text)
        doc.setTextWidth(max(300, options.rect.width()))
        return QtCore.QSize(doc.idealWidth(), doc.size().height())


class ComboBoxDelegate(QtGui.QItemDelegate):

    def __init__(self, list, parent, editable=True):
        QtGui.QItemDelegate.__init__(self, parent)
        self.list = list
        self.editable = editable

    def createEditor(self, parent, option, index):
        combo = QtGui.QComboBox(parent)
        combo.setEditable(self.editable)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
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
    #     hint = super(ComboBoxDelegate, self).sizeHint(option, index)
    #     style = self.parent().style()
    #     opt = QtGui.QStyleOptionComboBox()
    #     opt.initFrom(self.parent())
    #     rc = style.subControlRect(QtGui.QStyle.CC_ComboBox, opt, QtGui.QStyle.SC_ComboBoxArrow, self.parent())
    #     try:
    #         lst = list(self.list())
    #     except TypeError:
    #         lst = self.list
    #     fm = option.fontMetrics
    #     width = max(fm.width(l) for l in lst)
    #     hint.setWidth(width + rc.width() + 4)
    #     print hint.width()

    def eventFilter(self, combo, event):
        if not self.editable and event.type() == QtCore.QEvent.Enter and self._first_enter:
            combo.showPopup()
            self._first_enter = False
            return True
        else:
            return super(ComboBoxDelegate, self).eventFilter(combo, event)


class CheckBoxDelegate(QtGui.QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        """
        Important, otherwise an editor is created if the user clicks in this cell.
        """
        return None

    def paint(self, painter, option, index):
        """
        Paint a checkbox without the label.
        """
        if not index.model().data(index, Qt.UserRole):
            return

        checked = bool(index.model().data(index, Qt.DisplayRole))
        check_box_style_option = QtGui.QStyleOptionButton()

        if index.flags() & Qt.ItemIsEditable:
            check_box_style_option.state |= QtGui.QStyle.State_Enabled
        else:
            check_box_style_option.state |= QtGui.QStyle.State_ReadOnly

        if checked:
            check_box_style_option.state |= QtGui.QStyle.State_On
        else:
            check_box_style_option.state |= QtGui.QStyle.State_Off

        check_box_style_option.rect = self.get_check_box_rect(option)
        if not (index.model().flags(index) & Qt.ItemIsEditable):
            check_box_style_option.state |= QtGui.QStyle.State_ReadOnly

        QtGui.QApplication.style().drawControl(QtGui.QStyle.CE_CheckBox, check_box_style_option, painter)

    def editorEvent(self, event, model, option, index):
        """
        Change the data in the model and the state of the checkbox
        if the user presses the left mousebutton or presses
        Key_Space or Key_Select and this cell is editable. Otherwise do nothing.
        """
        if not (index.flags() & Qt.ItemIsEditable):
            return False

        # Do not change the checkbox-state
        if event.type() == QtCore.QEvent.MouseButtonRelease or event.type() == QtCore.QEvent.MouseButtonDblClick:
            if event.button() != Qt.LeftButton or not self.get_check_box_rect(option).contains(event.pos()):
                return False
            if event.type() == QtCore.QEvent.MouseButtonDblClick:
                return True
        elif event.type() == QtCore.QEvent.KeyPress:
            if event.key() != Qt.Key_Space and event.key() != Qt.Key_Select:
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
        new_value = not bool(index.model().data(index, Qt.DisplayRole))
        model.setData(index, new_value, Qt.EditRole)

    @staticmethod
    def get_check_box_rect(option):
        check_box_style_option = QtGui.QStyleOptionButton()
        check_box_rect = QtGui.QApplication.style().subElementRect(QtGui.QStyle.SE_CheckBoxIndicator,
                                                                   check_box_style_option, None)
        check_box_point = QtCore.QPoint (option.rect.x() +
                                         option.rect.width() / 2 -
                                         check_box_rect.width() / 2,
                                         option.rect.y() +
                                         option.rect.height() / 2 -
                                         check_box_rect.height() / 2)
        return QtCore.QRect(check_box_point, check_box_rect.size())


class VerticalScrollArea(QtGui.QScrollArea):

    def __init__(self, parent=None):
        super(VerticalScrollArea, self).__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def resizeEvent(self, event):
        super(VerticalScrollArea, self).resizeEvent(event)
        widget = self.widget()
        if widget is not None:
            widget.setFixedWidth(event.size().width())

    def eventFilter(self, obj, event):
        if obj and obj == self.widget() and event.type() == QtCore.QEvent.Resize:
            self.setMinimumWidth(obj.minimumSizeHint().width() + self.verticalScrollBar().width())
        return super(VerticalScrollArea, self).eventFilter(obj, event)


class ComboBox(QtGui.QComboBox):

    editingFinished = QtSignal()

    #Please do not mix this two signals!
    #currentIndexChanged is also emitted from very unexpected places like addItems, which ruins some my code (in Solver undo)
    #Instead, just connect to both: editingFinished and currentIndexChanged
    #PB
    #def __init__(self, *args, **kwargs):
    #    super(ComboBox, self).__init__(*args, **kwargs)
    #    self.currentIndexChanged.connect(lambda *args: self.editingFinished.emit())

    def focusOutEvent(self, event):
        if not self.signalsBlocked(): self.editingFinished.emit()
        super(ComboBox, self).focusOutEvent(event)

    def keyPressEvent(self, event):
        super(ComboBox, self).keyPressEvent(event)
        if event.key() in (Qt.Key_Enter, Qt.Key_Return) and not self.signalsBlocked():
            self.editingFinished.emit()


class InfoListView(QtGui.QListView):

    def __init__(self, model, parent):
        super(InfoListView, self).__init__(parent)
        self.model = model
        self.setMouseTracking(True)
        self._orig_height = None

    def enterEvent(self, event):
        super(InfoListView, self).enterEvent(event)
        rows = self.model.rowCount()
        if rows > 0:
            self._orig_height = self.height()
            self.setFixedHeight(self.sizeHintForRow(0) * rows)

    def leaveEvent(self, event):
        super(InfoListView, self).leaveEvent(event)
        if self._orig_height is not None:
            self.setFixedHeight(self._orig_height)
            self._orig_height = None


class TextEdit(QtGui.QTextEdit):
    editingFinished = QtSignal()

    def focusOutEvent(self, event):
        if not self.signalsBlocked(): self.editingFinished.emit()
        super(TextEdit, self).focusOutEvent(event)

    def keyPressEvent(self, event):
        super(TextEdit, self).keyPressEvent(event)
        if event.key() in (Qt.Key_Enter, Qt.Key_Return) and event.modifiers() == Qt.CTRL and not self.signalsBlocked():
            self.editingFinished.emit()


class TextEditWithCB(QtGui.QPlainTextEdit):
    """
        QPlainTextEdit which emits some extra callbacks:
        focus_out_cb - when it lost focus
        key_cb - when kay is pressed
    """

    def __init__(self, focus_out_cb=None, key_cb=None, **kwargs):
        super(TextEditWithCB, self).__init__(**kwargs)
        self.focus_out_cb = focus_out_cb
        self.key_cb = key_cb

    def focusOutEvent(self, event):
        super(TextEditWithCB, self).focusOutEvent(event)
        if self.focus_out_cb is not None: self.focus_out_cb()

    def keyPressEvent(self, event):
        super(TextEditWithCB, self).keyPressEvent(event)
        if self.key_cb is not None: self.key_cb(event)


class MultiLineEdit(QtGui.QWidget):
    """
    Widget showing multiple lines
    """

    def __init__(self, movable=False, change_cb=None, placeholder=None):
        super(MultiLineEdit, self).__init__()
        self.change_cb = change_cb
        self.placeholder = '[enter value]' if placeholder is None else placeholder
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.list_widget = QtGui.QListWidget()
        layout.addWidget(self.list_widget)
        buttons = QtGui.QVBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(1)
        add = QtGui.QToolButton()
        add.setIcon(QtGui.QIcon.fromTheme('list-add'))
        add.pressed.connect(self.add_item)
        buttons.addWidget(add)
        act = QtGui.QAction(self.list_widget)
        act.setShortcut(QtGui.QKeySequence(Qt.Key_Plus))
        act.triggered.connect(self.add_item)
        act.setShortcutContext(Qt.WidgetShortcut)
        self.list_widget.addAction(act)
        self.edit = QtGui.QToolButton()
        self.edit.setIcon(QtGui.QIcon.fromTheme('document-edit'))
        self.edit.pressed.connect(self.edit_item)
        buttons.addWidget(self.edit)
        self.remove = QtGui.QToolButton()
        self.remove.setIcon(QtGui.QIcon.fromTheme('list-remove'))
        self.remove.pressed.connect(self.remove_item)
        buttons.addWidget(self.remove)
        act = QtGui.QAction(self.list_widget)
        act.setShortcut(QtGui.QKeySequence(Qt.Key_Delete))
        act.triggered.connect(self.remove_item)
        act.setShortcutContext(Qt.WidgetShortcut)
        self.list_widget.addAction(act)
        self._movable = movable
        if movable:
            self.up = QtGui.QToolButton()
            self.up.setIcon(QtGui.QIcon.fromTheme('go-up'))
            self.up.pressed.connect(self.move_up)
            act = QtGui.QAction(self.list_widget)
            act.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_Up)
            act.triggered.connect(self.move_up)
            act.setShortcutContext(Qt.WidgetShortcut)
            self.list_widget.addAction(act)
            buttons.addWidget(self.up)
            self.down = QtGui.QToolButton()
            self.down.setIcon(QtGui.QIcon.fromTheme('go-down'))
            self.down.pressed.connect(self.move_down)
            buttons.addWidget(self.down)
            act = QtGui.QAction(self.list_widget)
            act.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_Down)
            act.triggered.connect(self.move_down)
            act.setShortcutContext(Qt.WidgetShortcut)
            self.list_widget.addAction(act)
        else:
            self.up = None
            self.down = None
        layout.addLayout(buttons)
        self.selected()
        self.list_widget.itemSelectionChanged.connect(self.selected)
        self.list_widget.itemChanged.connect(self.item_changed)

    def selected(self):
        row = self.list_widget.currentRow()
        if row != -1:
            self.edit.setEnabled(True)
            self.remove.setEnabled(True)
            if self.up is not None:
                self.up.setEnabled(row > 0)
            if self.down is not None:
                self.down.setEnabled(row < self.list_widget.count()-1)
        else:
            self.edit.setEnabled(False)
            self.remove.setEnabled(False)
            if self.up is not None: self.up.setEnabled(False)
            if self.down is not None: self.down.setEnabled(False)

    def item_changed(self, item):
        if item is not None:
            self.list_widget.setCurrentItem(item)
        if self.change_cb is not None:
            self.change_cb()

    def add_item(self):
        with BlockQtSignals(self.list_widget):
            item = QtGui.QListWidgetItem(self.placeholder)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            current = self.list_widget.currentRow()
            if self._movable and current != -1:
                self.list_widget.insertItem(current+1, item)
            else:
                self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(None)
        self.list_widget.scrollToItem(item)
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
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
            else:
                for i in range(n):
                    item = self.list_widget.item(i)
                    item.setText(values[i])

    def get_values(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())]


def fire_edit_end(widget=None):
    """
    Try to call event which cause updating model by widget which is focused (or widget given as parameter).
    :param QtGui.QWidget widget: QtGui.QApplication.focusWidget() will be used by default
    """
    if widget is None: widget = QtGui.QApplication.focusWidget()
    try: widget.editingFinished.emit()
    except: pass
    try: widget.focus_out_cb()
    except: pass

