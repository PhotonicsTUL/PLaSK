from PyQt4 import QtCore, QtGui
import weakref

class Info(object):
    
    INFO = 0
    WARNING = 1
    ERROR = 2
    
    def __str__(self):
        return self.text
    
    def __init__(self, text, level = None, **kwargs):
        object.__init__(self)
        self.text = text
        self.level = int(level)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
    def add_connection(self, attr_name, value):
        getattr(self, attr_name, []).append(value)
               
    def has_connection(self, attr_name, value, ans_if_non_attr = True):
        """
            Check if self has attribute with given name which includes given value.
            For example: self.has_connection('rows', 5)
            :param attr_name: required attribute
            :param value: required value
            :param ans_if_non_attr: result which is returned when object has no required attribute
            :return: True if self has attribute with given name which includes given value,
                     False if self has attribute with given name which dpesn't include given value,
                     ans_if_non_attr if self has not attribute with given name.
        """
        #if hasattr(self, attr_name): return getattr(self, attr_name) == value
        if hasattr(self, attr_name): return value in getattr(self, attr_name)   # + 's'
        return ans_if_non_attr
    
def infoLevelIcon(level):
    if level == Info.INFO: return QtGui.QIcon.fromTheme('dialog-information')
    if level == Info.WARNING: return QtGui.QIcon.fromTheme('dialog-warning')
    if level == Info.ERROR: return QtGui.QIcon.fromTheme('dialog-error')
    return None

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
            self.entries = model.info
            model.infoChanged += self.infoChanged
    
    def __init__(self, model, parent=None, *args):
        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self.__setModel__(model)
                       
    def infoChanged(self, model):
        """Read info from model, inform observers."""
        self.layoutAboutToBeChanged.emit()
        self.entries = model.info
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
            return infoLevelIcon(self.entries[index.row()].level)
        return None
    
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return 'text'
        return None