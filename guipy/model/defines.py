from model.base import SectionModel
from PyQt4 import QtCore
from xml.etree import ElementTree
#from guis import DefinesEditor

class DefinesModel(QtCore.QAbstractTableModel, SectionModel):
    
    class Entry:
        def __init__(self, name, value, comment = None):
            self.name = name
            self.value = value
            self.comment = comment
    
    def __init__(self, parent=None, *args):
        SectionModel.__init__(self, 'defines')
        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self.defines = []
        
    def nameToIndex(self, name):
        """return index of entry with given name or -1"""
        for idx, val in enumerate(self.defines):
            if val.name == name: return idx
        return -1
        
    def setXMLElement(self, element):
        self.layoutAboutToBeChanged.emit()
        del self.defines[:]
        if isinstance(element, ElementTree.Element):
            for c in element.iter("define"):
                self.defines.append(DefinesModel.Entry(c.attrib["name"], c.attrib["value"]))
        self.layoutChanged.emit()
    
    # XML element that represents whole section
    def getXMLElement(self):
        res = ElementTree.Element(self.tagname)
        for e in self.defines:
            ElementTree.SubElement(res, "define", { "name": e.name, "value": e.value }).tail = '\n'
        return res
    
    def get(self, col, row):
        if col == 0: return self.defines[row].name
        if col == 1: return self.defines[row].value
        if col == 2: return self.defines[row].comment
        raise IndexError('column number for DefinesModel should be in range [0, 3], but is %d' % col)
    
    def set(self, col, row, value):
        if col == 0:
            i = self.nameToIndex(value)
            if i > 0 and i != row: raise ValueError("name \"%s\" already in use in defines section (has indexes %d and value \"%s\")" % (value, i, self.defines[i].value))
            self.defines[row].name = value
        elif col == 1: self.defines[row].value = value
        elif col == 2: self.defines[row].comment = value
        else: raise IndexError('column number for DefinesModel should be in range [0, 3], but is %d' % col)       
        
    # QAbstractListModel implementation
    def rowCount(self, parent = QtCore.QModelIndex()):
        if parent.isValid(): return 0
        return len(self.defines)
    
    def columnCount(self, parent = QtCore.QModelIndex()): 
        return 2    # 3 if comment supported
    
    def data(self, index, role = QtCore.Qt.DisplayRole): 
        if index.isValid() and (role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole): 
            return self.get(index.column(), index.row())
        return None
        
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'name'
            if col == 1: return 'value'
            if col == 2: return 'comment'
        return None

    def flags(self, index):
        flags = super(self.__class__, self).flags(index)

        flags |= QtCore.Qt.ItemIsEditable
        flags |= QtCore.Qt.ItemIsSelectable
        flags |= QtCore.Qt.ItemIsEnabled
        flags |= QtCore.Qt.ItemIsDragEnabled
        flags |= QtCore.Qt.ItemIsDropEnabled

        return flags
    
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        self.set(index.column(), index.row(), value)
        self.dataChanged.emit(index, index)
        return True