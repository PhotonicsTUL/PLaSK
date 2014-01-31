from model.defines import DefinesModel
from controler.table import TableControler
from PyQt4 import QtGui, QtCore

class AfterBracketCompleter(QtGui.QCompleter):
        
    def pathFromIndex(self, index):
        path = QtGui.QCompleter.pathFromIndex(self, index)
        
        lst = str(self.widget().text()).rsplit('{', 1)
        if len(lst) > 1:
            path = '%s{%s}' % (lst[0], path)
        else:
            path = '{%s}' % path
    
        return path
    
    def splitPath(self, path):
        path = path.rsplit('{', 1)[-1].lstrip(' ')
        return [path]
        
class DefineHinstTableModel(QtCore.QAbstractTableModel):
   
    def __init__(self, defineModel, parent = None, info_cb = None, *args):
        QtCore.QAbstractListModel.__init__(self, parent, *args)   #QtCore.QObject.parent(defineModel)
        self.model = defineModel     
        
    def rowCount(self, parent = QtCore.QModelIndex()):
        return self.model.rowCount(parent)
    
    def data(self, index, role = QtCore.Qt.DisplayRole):
        if index.isValid() and index.column() == 1:
            if role == QtCore.Qt.FontRole: 
                font = QtGui.QFont()
                font.setItalic(True)
                return font
            if role == QtCore.Qt.TextColorRole:
                return QtGui.QColor(90, 90, 90) #QtGui.QColor(QtCore.Qt.blue)
        return self.model.data(index, role) 
        
    #def flags(self, index):
    #    return super(DefineHinstTableModel, self).flags(index) | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
    
    def columnCount(self, parent = QtCore.QModelIndex()): 
        return 2
            
    def headerData(self, col, orientation, role):
        return self.model.headerData(col, orientation, role)
        
class DefinesCompletionDelegate(QtGui.QItemDelegate):

    def __init__(self, model, parent):
        QtGui.QItemDelegate.__init__(self, parent)
        self.model = DefineHinstTableModel(model, parent)
        #self.model = model
        
    def createEditor(self, parent, option, index):
        ed = super(DefinesCompletionDelegate, self).createEditor(parent, option, index)
        completer = AfterBracketCompleter(self.model, self)
        tab = QtGui.QTableView(parent)
        #tab.resizeColumnsToContents()
        tab.setModel(self.model)
        tab.setMinimumSize(0, 200)
        #tab.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        tab.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        tab.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        tab.setSelectionBehavior(QtGui.QTableView.SelectRows)
        tab.setSelectionMode(QtGui.QTableView.SingleSelection)
        tab.horizontalHeader().hide()
        tab.verticalHeader().hide()
        tab.setSortingEnabled(False)
        tab.setShowGrid(False)
        tab.setWordWrap(False)
        #tab.setContentsMargins(1, 1, 1, 1)
        
        completer.setPopup(tab)
        #completer.setWrapAround(False)
        #completer->setCaseSensitivity(Qt::CaseInsensitive);
        ed.setCompleter(completer)
        return ed
        
    #def setEditorData(self, editor, index):
    #    editor.blockSignals(True)
    #    editor.setCurrentIndex(int(index.model().data(index)))
    #    editor.blockSignals(False)
        
    #def setModelData(self, editor, model, index):
    #    model.setData(index, editor.currentIndex())
        
    #@QtCore.pyqtSlot()
    #def currentIndexChanged(self):
    #    self.commitData.emit(self.sender())

class DefinesControler(TableControler):

    def __init__(self, document, model = DefinesModel()):
        TableControler.__init__(self, document, model)
        self.table.setItemDelegateForColumn(1, DefinesCompletionDelegate(self.model, self.table))
