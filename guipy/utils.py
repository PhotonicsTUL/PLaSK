# needs by Signal:
from __future__ import print_function
import inspect
from weakref import WeakSet, WeakKeyDictionary


from PyQt4 import QtCore 
from PyQt4 import QtGui
import collections

def exceptionToMsg(f, parent = None, err_title = None):   
    try:
        f()
        return True
    except Exception as e:
        QtGui.QMessageBox().critical(parent, err_title, str(e))
        return False

#deprecated, to remove
def toPythonStr(qstr):
    if isinstance(qstr, QtCore.QString):
        return unicode (qstr.toUtf8(), "utf-8")
    else:
        return qstr
    
defaultFont = QtGui.QFont()
defaultFont.setFamily("Courier")
defaultFont.setPointSize(10)

""" A signal/slot implementation + __iadd__ and __isub__ (Piotr Beling)

Author:  Thiago Marcos P. Santos
Author:  Christopher S. Case
Author:  David H. Bronke
Created: August 28, 2008
Updated: December 12, 2011
License: MIT

"""
class Signal(object):
    def __init__(self):
        self._functions = WeakSet()
        self._methods = WeakKeyDictionary()

    def __call__(self, *args, **kargs):
        # Call handler functions
        for func in self._functions:
            func(*args, **kargs)

        # Call handler methods
        for obj, funcs in self._methods.items():
            for func in funcs:
                func(obj, *args, **kargs)

    def connect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ not in self._methods:
                self._methods[slot.__self__] = set()

            self._methods[slot.__self__].add(slot.__func__)

        else:
            self._functions.add(slot)
            
    def __iadd__(self, slot):
        self.connect(slot)
        return self

    def disconnect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ in self._methods:
                self._methods[slot.__self__].remove(slot.__func__)
        else:
            if slot in self._functions:
                self._functions.remove(slot)
                
    def __isub__(self, slot):
        self.disconnect(slot)
        return self

    def clear(self):
        self._functions.clear()
        self._methods.clear()
        
def table_last_col_fill(table, cols_count, col_size = 200):
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
        doc.setHtml(options.text)
        doc.setTextWidth(max(300, options.rect.width()))    #TODO 300 -> member

        options.text = ""
        style.drawControl(QtGui.QStyle.CE_ItemViewItem, options, painter);

        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()

        # Highlighting text if item is selected
        #if (optionV4.state & QStyle::State_Selected)
            #ctx.palette.setColor(QPalette::Text, optionV4.palette.color(QPalette::Active, QPalette::HighlightedText));

        textRect = style.subElementRect(QtGui.QStyle.SE_ItemViewItemText, options)
        painter.save()
        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        doc.documentLayout().draw(painter, ctx)

        painter.restore()

    def sizeHint(self, option, index):
        options = QtGui.QStyleOptionViewItemV4(option)
        self.initStyleOption(options, index)

        doc = QtGui.QTextDocument()
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