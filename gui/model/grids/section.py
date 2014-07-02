from lxml import etree as ElementTree
from ...qt import QtCore

from ...controller.grids.new_dialog import construct_grid_using_dialog
from ..table import TableModel
from .types import construct_grid


class GridsModel(TableModel):

    def __init__(self, parent=None, info_cb=None, *args):
        super(GridsModel, self).__init__('grids', parent, info_cb, *args)

    def set_XML_element(self, element):
        self.layoutAboutToBeChanged.emit()
        del self.entries[:]
        if element is not None:
            for g in element:
                self.entries.append(construct_grid(self, g))
        self.layoutChanged.emit()
        self.fire_changed()

    # XML element that represents whole section
    def get_XML_element(self):
        res = ElementTree.Element(self.name)
        for e in self.entries: res.append(e.get_XML_element())
        return res

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2    # 3 if comment supported

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'Name'
            if col == 1: return 'Type (Method)'
            if col == 2: return 'Comment'
        return None

    def get(self, col, row):
        if col == 0: return self.entries[row].name
        if col == 1: return self.entries[row].type_and_kind_str
        if col == 2: return self.entries[row].comment
        raise IndexError('column number for GridsModel should be 0, 1, or 2, but is %d' % col)

    def set(self, col, row, value):
        if col == 0: self.entries[row].name = value
        #else: raise IndexError('column number for MaterialsModel should be 0, 1, or 2, but is %d' % col)

    def flags(self, index):
        flags = super(GridsModel, self).flags(index)
        if index.column() == 1: flags &= ~QtCore.Qt.ItemIsEditable
        return flags

    def create_default_entry(self):
        return construct_grid_using_dialog(self)
