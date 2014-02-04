from PyQt4 import QtCore
from xml.etree import ElementTree
from model.table import TableModel
from collections import OrderedDict
from model.info import Info
#from guis import DefinesEditor

class DefinesModel(TableModel):
    
    class Entry:
        def __init__(self, name, value, comment = None):
            self.name = name
            self.value = value
            self.comment = comment
    
    def __init__(self, parent=None, info_cb = None, *args):
        TableModel.__init__(self, 'defines', parent, info_cb, *args)
        
    def nameToIndex(self, name):
        """return index of entry with given name or -1"""
        for idx, val in enumerate(self.entries):
            if val.name == name: return idx
        return -1
        
    def setXMLElement(self, element):
        self.layoutAboutToBeChanged.emit()
        del self.entries[:]
        if isinstance(element, ElementTree.Element):
            for c in element.iter("define"):
                self.entries.append(DefinesModel.Entry(c.attrib["name"], c.attrib["value"]))
        self.layoutChanged.emit()
        self.fireChanged()
    
    # XML element that represents whole section
    def getXMLElement(self):
        res = ElementTree.Element(self.name)
        for e in self.entries:
            ElementTree.SubElement(res, "define", { "name": e.name, "value": e.value }).tail = '\n'
        return res
    
    def get(self, col, row):
        if col == 0: return self.entries[row].name
        if col == 1: return self.entries[row].value
        if col == 2: return self.entries[row].comment
        raise IndexError('column number for DefinesModel should be 0, 1, or 2, but is %d' % col)
    
    def set(self, col, row, value):
        if col == 0: self.entries[row].name = value
        elif col == 1: self.entries[row].value = value
        elif col == 2: self.entries[row].comment = value
        else: raise IndexError('column number for DefinesModel should be 0, 1, or 2, but is %d' % col)
        
    def createInfo(self):
        res = super(DefinesModel, self).createInfo()
        names = OrderedDict()
        for i, d in enumerate(self.entries):
            names.setdefault(d.name, []).append(i)
        for name, indexes in names.items():
            if len(indexes) > 1:
                err = Info('Duplicated definition name "%s", rows: %s' % (name, ', '.join(map(str, indexes))), Info.ERROR)
                err.rows = indexes
                err.cols = [0]
                res.append(err)
        return res
        
    def createDefaultEntry(self):
        return DefinesModel.Entry("new", "")
    
    # QAbstractListModel implementation
    
    def columnCount(self, parent = QtCore.QModelIndex()): 
        return 2    # 3 if comment supported
            
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'name'
            if col == 1: return 'value'
            if col == 2: return 'comment'
        return None