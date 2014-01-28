from PyQt4 import QtCore, QtGui
import weakref

class Info(object):
    
    INFO = 0
    WARNING = 1
    ERROR = 2
    
    def __str__(self):
        return self.text
    
    def __init__(self, text, level = None):
        object.__init__(self)
        self.text = text
        self.level = int(level)

class InfoListModel(QtCore.QAbstractListModel):
    """Qt list model of info (warning, errors, etc.) of section model (None section model is allowed and than the list is empty)"""
    
    def __setModel__(self, model):
        if hasattr(self, 'model'):
            m = self.model()
            if m: m.infoChanged -= self.infoChanged
        if model == None:
            if hasattr(self, 'model'): del self.model
            self.entries = []
        else:
            self.model = weakref.ref(model)
            self.entries = model.getInfo()
            model.infoChanged += self.infoChanged
    
    def __init__(self, model, parent=None, *args):
        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self.__setModel__(model)
                       
    def infoChanged(self, model):
        """Read info from model, inform observers."""
        self.layoutAboutToBeChanged.emit()
        self.entries = model.getInfo()
        self.layoutChanged.emit()
        
    def setModel(self, model):
        self.__setModel__(model)
        if model != None: self.infoChanged(model)
        
    def rowCount(self, parent = QtCore.QModelIndex()):
        if parent.isValid(): return 0
        return len(self.entries)
    
    #def columnCount(self, parent = QtCore.QModelIndex()): 
    #    return 1
    
    def data(self, index, role = QtCore.Qt.DisplayRole): 
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole: 
            return self.entries[index.row()].text
        if role == QtCore.Qt.DecorationRole:
            l = self.entries[index.row()].level
            if l == Info.INFO: return QtGui.QIcon.fromTheme('dialog-information')
            if l == Info.WARNING: return QtGui.QIcon.fromTheme('dialog-warning')
            if l == Info.ERROR: return QtGui.QIcon.fromTheme('dialog-error')
        return None
    
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return 'text'
        return None