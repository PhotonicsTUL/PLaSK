from ..qt import QtCore
from collections import OrderedDict
from lxml import etree as ElementTree

from .table import TableModel
from .info import Info
#from guis import DefinesEditor


class DefinesModel(TableModel):

    class Entry:
        def __init__(self, name, value, comment=None):
            self.name = name
            self.value = value
            self.comment = comment

    def __init__(self, parent=None, info_cb = None, *args):
        TableModel.__init__(self, 'defines', parent, info_cb, *args)

    def nameToIndex(self, name):
        """:return: index of entry with given name or -1"""
        for idx, val in enumerate(self.entries):
            if val.name == name: return idx
        return -1

    def set_XML_element(self, element):
        self.modelAboutToBeReset.emit()
        if element is not None:
            self.entries = [DefinesModel.Entry(c.attrib.get("name", ""), c.attrib.get("value", ""))
                            for c in element.iter("define")]
        else:
            self.entries = []
        self.modelReset.emit()
        self.fire_changed()

    # XML element that represents whole section
    def get_XML_element(self):
        res = ElementTree.Element(self.name)
        for e in self.entries:
            if e.comment: res.append(ElementTree.Comment(e.comment))
            ElementTree.SubElement(res, "define", { "name": e.name, "value": e.value }) #.tail = '\n'
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

    def create_info(self):
        res = super(DefinesModel, self).create_info()
        names = OrderedDict()
        for i, d in enumerate(self.entries):
            if not d.name:
                res.append(Info('Definition name is required [row: %d]' % i, Info.ERROR, rows = [i], cols = [0]))
            else:
                names.setdefault(d.name, []).append(i)
            if not d.value: res.append(Info('Definition value is required [row: %d]' % i, Info.ERROR, rows = [i], cols = [1]))
        for name, indexes in names.items():
            if len(indexes) > 1:
                res.append(Info('Duplicated definition name "%s" [rows: %s]' % (name, ', '.join(map(str, indexes))),
                                Info.ERROR, rows = indexes, cols = [0]
                                )
                          )
        return res

    def create_default_entry(self):
        return DefinesModel.Entry("new", "")

    # QAbstractListModel implementation

    def columnCount(self, parent = QtCore.QModelIndex()):
        return 2    # 3 if comment supported

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'Name'
            if col == 1: return 'Value'
            if col == 2: return 'Comment'
        return None

    #zapis DEF.cos nie dziala w PLaSKu
    #def stubs(self):
        #res = "DEF = object()\n"
        #res += "\n".join("DEF."%s = 0" % d.name for d in self.entries)
        #return res
