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
import sys

from ..qt import QtCore
from ..qt import QtGui
from ..utils.config import CONFIG

def exception_to_msg(f, parent=None, err_title=None):
    """
        Call f() in try block, and after catch exception show Qt message.
        :return: false only if F() has thrown an exception, true in other cases (bool)
    """
    try:
        f()
        return True
    except Exception as e:
        from .. import _DEBUG
        if _DEBUG:
            import traceback as tb
            tb.print_exc()
        else:
            QtGui.QMessageBox().critical(parent, err_title, str(e))
        return False


DEFAULT_FONT = QtGui.QFont()
_font_family = CONFIG['editor/font_family']
if _font_family is None:
    if sys.platform == 'win32':
        _font_family = "Consolas"
    elif sys.platform == 'darwin':
        _font_family = "Monaco"
    else:
        _font_family = "Monospace"
    CONFIG['editor/font_family'] = _font_family
    DEFAULT_FONT.setStyleHint(QtGui.QFont.TypeWriter)
DEFAULT_FONT.setFamily(_font_family)
del _font_family
DEFAULT_FONT.setPointSize(int(CONFIG('editor/font_size', DEFAULT_FONT.pointSize())))


def table_last_col_fill(table, cols_count, col_size=0):
    if isinstance(col_size, collections.Sequence):
        for c in range(0, cols_count-1): table.setColumnWidth(c, col_size[c])
    else:
        for c in range(0, cols_count-1): table.setColumnWidth(c, col_size)
    #table.horizontalHeader().setResizeMode(cols_count-1, QtGui.QHeaderView.Stretch)
    table.horizontalHeader().setStretchLastSection(True)


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


class ComboBoxDelegate(QtGui.QStyledItemDelegate):

    def __init__(self, list, parent):
        QtGui.QItemDelegate.__init__(self, parent)
        self.list = list
        #self.model = model

    def createEditor(self, parent, option, index):
        combo = QtGui.QComboBox(parent)
        combo.setEditable(True)
        combo.setInsertPolicy(QtGui.QComboBox.NoInsert)
        combo.addItems(self.list)
        #self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"),
        #             self, QtCore.SLOT("currentIndexChanged()"))
        return combo


#class CheckBoxDelegate(QtGui.QStyledItemDelegate):

    #def createEditor(self, parent, option, index):
        #'''
        #Important, otherwise an editor is created if the user clicks in this cell.
        #'''
        #return None

    #def paint(self, painter, option, index):
        #'''
        #Paint a checkbox without the label.
        #'''
        #checked = bool(index.model().data(index, QtCore.Qt.DisplayRole))
        #check_box_style_option = QtGui.QStyleOptionButton()

        #if (index.flags() & QtCore.Qt.ItemIsEditable) > 0:
            #check_box_style_option.state |= QtGui.QStyle.State_Enabled
        #else:
            #check_box_style_option.state |= QtGui.QStyle.State_ReadOnly

        #if checked:
            #check_box_style_option.state |= QtGui.QStyle.State_On
        #else:
            #check_box_style_option.state |= QtGui.QStyle.State_Off

        #check_box_style_option.rect = self.getCheckBoxRect(option)

        #QtGui.QApplication.style().drawControl(QtGui.QStyle.CE_CheckBox, check_box_style_option, painter)


    #def editorEvent(self, event, model, option, index):
        #'''
        #Change the data in the model and the state of the checkbox
        #if the user presses the left mousebutton or presses
        #Key_Space or Key_Select and this cell is editable. Otherwise do nothing.
        #'''
        #if not (index.flags() & QtCore.Qt.ItemIsEditable) > 0:
            #return False

        ## Do not change the checkbox-state
        #if event.type() == QtGui.QEvent.MouseButtonRelease or event.type() == QtCore.QEvent.MouseButtonDblClick:
            #if event.button() != QtCore.Qt.LeftButton or not self.getCheckBoxRect(option).contains(event.pos()):
                #return False
            #if event.type() == QtCore.QEvent.MouseButtonDblClick:
                #return True
        #elif event.type() == QtCore.QEvent.KeyPress:
            #if event.key() != QtCore.Qt.Key_Space and event.key() != QtCore.Qt.Key_Select:
                #return False
        #else:
            #return False

        ## Change the checkbox-state
        #self.setModelData(None, model, index)
        #return True

    #def setModelData (self, editor, model, index):
        #'''
        #The user wanted to change the old state in the opposite.
        #'''
        #newValue = not bool(index.model().data(index, QtCore.Qt.DisplayRole))
        #model.setData(index, newValue, QtCore.Qt.EditRole)


    #def getCheckBoxRect(self, option):
        #check_box_style_option = QtGui.QStyleOptionButton()
        #check_box_rect = QtGui.QApplication.style().subElementRect(QtGui.QStyle.SE_CheckBoxIndicator,
                                                                   #check_box_style_option, None)
        #check_box_point = QtCore.QPoint (option.rect.x() +
                             #option.rect.width() / 2 -
                             #check_box_rect.width() / 2,
                             #option.rect.y() +
                             #option.rect.height() / 2 -
                             #check_box_rect.height() / 2)
        #return QtCore.QRect(check_box_point, check_box_rect.size())
