from PyQt4 import QtCore 
from PyQt4 import QtGui
import collections

def exception_to_msg(f, parent = None, err_title = None):
    """
        Call f() in try block, and after catch exception show Qt message.
        :return: false obly if F() has thrown an exception, true in other cases (bool)
    """   
    try:
        f()
        return True
    except Exception as e:
        QtGui.QMessageBox().critical(parent, err_title, str(e))
        return False
    
defaultFont = QtGui.QFont()
defaultFont.setFamily("Courier")
defaultFont.setPointSize(10)

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