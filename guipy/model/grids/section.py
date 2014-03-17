from model.table import TableModel
from model.grids.grid import GridTreeBased
from model.grids.mesh_rectilinear import RectilinearMesh
from lxml import etree as ElementTree
from PyQt4 import QtCore

def contruct_mesh(element):
    t = element.attrib['type']
    if t in ['rectilinear1d', 'rectilinear2d', 'rectilinear3d']: return RectilinearMesh.from_XML(element)
    return GridTreeBased.from_XML(element)

def contruct_generator(element):
    return GridTreeBased.from_XML(element)


def contruct_grid(element):
    
    if element.tag == "mesh":
        k = element.attrib.keys()
        k.sort()
        if k != ['name', 'type']: raise ValueError('<mesh> tag must have two attributes (name and type), but has: %s' % ', '.join(k))
        return contruct_mesh(element)
    
    if element.tag == "generator":
        k = element.attrib.keys()
        k.sort()
        if k != ['method', 'name', 'type']: raise ValueError('<generator> tag must have attributes "method", "name" and "type", but has: %s' % ', '.join(k))
        return contruct_generator(element)
    
    raise ValueError('In <grids> section only <mesh> and <generator> tags are allowed, but got "%s".' % element.tag)


class GridsModel(TableModel):
    
    def __init__(self, parent=None, info_cb = None, *args):
        super(GridsModel, self).__init__('grids', parent, info_cb, *args)
        
    def setXMLElement(self, element):
        self.layoutAboutToBeChanged.emit()
        del self.entries[:]
        if element is not None:
            for g in element:
                self.entries.append(contruct_grid(g))
        self.layoutChanged.emit()
        self.fireChanged()
        
    # XML element that represents whole section
    def getXMLElement(self):
        res = ElementTree.Element(self.name)
        for e in self.entries: res.append(e.getXMLElement())
        return res
    
    def columnCount(self, parent = QtCore.QModelIndex()): 
        return 2    # 3 if comment supported
    
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'name'
            if col == 1: return 'type (and method)'
            if col == 2: return 'comment'
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