from base import *
from PyQt4 import QtCore
#from guis import DefinesEditor

class DefinesModel(SectionModel, QtCore.QAbstractTableModel):
    
    class Entry:
        def __init__(self, name, value, comment = None):
            self.name = name
            self.value = value
            self.comment = comment
    
    def __init__(self, parent=None, *args):
        SectionModel.__init__(self, 'defines')
        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self.defines = []
        
    def setXMLElement(self, element):
        del self.defines[:]
        if isinstance(element, ElementTree.Element):
            for c in element.iter("define"):
                self.defines.append(DefinesModel.Entry(c.attrib["name"], c.attrib["value"]))
    
    # XML element that represents whole section
    def getXMLElement(self):
        res = ElementTree.Element('defines')
        for e in self.defines:
            ElementTree.SubElement(res, "define", { "name": e.name, "value": e.value }).tail = '\n'
        return res
    
    def get(self, col, row):
        if col == 0: return self.defines[row].name
        if col == 1: return self.defines[row].value
        if col == 2: return self.defines[row].comment
        raise IndexError('column number for DefinesModel should be in range [0, 3], but is %d' % col)
    
    def set(self, col, row, value):
        if col == 0: self.defines[row].name = value
        elif col == 1: self.defines[row].value = value
        elif col == 2: self.defines[row].comment = value
        else: raise IndexError('column number for DefinesModel should be in range [0, 3], but is %d' % col)
    
    # QAbstractListModel implementation
    def rowCount(self, parent = QtCore.QModelIndex()):
        return len(self.defines)
    
    def columnCount(self, parent = QtCore.QModelIndex()): 
        return 2    # 3 if comment supported
    
    def data(self, index, role): 
        if index.isValid() and (role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole): 
            return self.get(index.column(), index.row())
        return QtCore.QVariant() 
        
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'name'
            if col == 1: return 'value'
            if col == 2: return 'comment'
        return QtCore.QVariant()

    def flags(self, index):
        flags = super(self.__class__, self).flags(index)

        flags |= QtCore.Qt.ItemIsEditable
        flags |= QtCore.Qt.ItemIsSelectable
        flags |= QtCore.Qt.ItemIsEnabled
        flags |= QtCore.Qt.ItemIsDragEnabled
        flags |= QtCore.Qt.ItemIsDropEnabled

        return flags
    
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        set(index.column(), index.row(), value)
        self.emit(QtCore.SIGNAL('dataChanged()'))
        return True