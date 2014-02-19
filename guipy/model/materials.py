# -*- coding: utf-8 -*-

from PyQt4 import QtCore
from xml.etree import ElementTree
from model.table import TableModel
from model.info import Info
from collections import OrderedDict
#from guis import DefinesEditor

MATERIALS_PROPERTES = {
    'A': ('Monomolecular recombination coefficient A [1/s]', [('T', 'temperature [K]')]),
    'absb': ('Absorption coefficient α [cm<sup>-1</sup>]', [('wl', 'wavelength [nm]'), ('T', 'temperature [K]')]),
    'ac': ('Hydrostatic deformation potential for the conduction band a<sub>c</sub> [eV]', [('T', 'temperature [K]')]),
    'av': ('Hydrostatic deformation potential for the valence band a<sub>v</sub> [eV]', [('T', 'temperature [K]')]),
    'B': ('Radiative recombination coefficient B [m<sup>3</sup>/s]', [('T', 'temperature [K]')]),
    'b': ('Shear deformation potential b [eV]', [('T', 'temperature [K]')]),
    'C': ('Auger recombination coefficient C [m<sup>6</sup>/s]', [('T', 'temperature [K]')]),
    'c11': ('Elastic constant c<sub>11</sub> [GPa]', [('T', 'temperature [K]')]),
    'c12': ('Elastic constant c<sub>12</sub> [GPa]', [('T', 'temperature [K]')]),
    'CB': ('Conduction band level CB [eV]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]'), ('point', 'point in the Brillouin zone [-]')]),
    'chi': ('Electron affinity χ [eV]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]'), ('point', 'point in the Brillouin zone [-]')]),
    'cond': ('Electrical conductivity sigma in-plane (lateral) and cross-plane (vertical) direction [S/m]', [('T', 'temperature [K]')]),
    'condtype': ('Electrical conductivity type. In semiconductors this indicates what type of carriers Nf refers to.', []),
    'cp': ('Specific heat at constant pressure [J/(kg K)]', [('T', 'temperature [K]')]),
    'D': ('Ambipolar diffusion coefficient D [m<sup>2</sup>/s]', [('T', 'temperature [K]')]),
    'dens': ('Density [kg/m<sup>3</sup>]', [('T', 'temperature [K]')]),
    'Dso': ('Split-off energy D<sub>so</sub> [eV]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]')]),
    'EactA': ('Acceptor ionization energy E<sub>actA</sub> [eV]', [('T', 'temperature [K]')]),
    'EactD': ('Acceptor ionization energy E<sub>actD</sub> [eV]', [('T', 'temperature [K]')]),
    'Eg': ('Energy gap E<sub>g</sub> [eV]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]'), ('point', 'point in the Brillouin')]),
    'eps': ('Donor ionization energy ε<sub>R</sub> [-]', [('T', 'temperature [K]')]),
    'lattC': ('Lattice constant [Å]', [('T', 'temperature [K]'), ('x', 'lattice parameter [-]')]),
    'Me': ('Electron effective mass M<sub>e</sub> in in-plane (lateral) and cross-plane (vertical) direction [m<sub>0</sub>]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]'), ('point', 'point in the irreducible Brillouin zone [-]')]),
    'Mh': ('Hole effective mass M<sub>h</sub> in in-plane (lateral) and cross-plane (vertical) direction [m<sub>0</sub>]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]')]),
    'Mhh': ('Heavy hole effective mass M<sub>hh</sub> in in-plane (lateral) and cross-plane (vertical) direction [m<sub>0</sub>]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]')]),
    'Mlh': ('Light hole effective mass M<sub>lh</sub> in in-plane (lateral) and cross-plane (vertical) direction [m<sub>0</sub>]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]')]),
    'Mso': ('Split-off mass M<sub>so</sub>` [m<sub>0</sub>]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]')]),
    'Nc': ('Effective density of states in the conduction band Nc [cm<sup>-3</sup>]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]'), ('point', 'point in the Brillouin zone [-]')]),
    'Nf': ('Free carrier concentration N [cm<sup>-3</sup>]', [('T', 'temperature [K]')]),
    'Ni': ('Intrinsic carrier concentration N<sub>i</sub> [cm<sup>-3</sup>]', [('T', 'temperature [K]')]),
    'Nr': ('Complex refractive index n<sub>R</sub> [-]', [('wl', 'wavelength [nm]'), ('T', 'temperature [K]'), ('n', 'injected carriers concentration [1/cm]')]), 
    'nr': ('Real refractive index n<sub>R</sub> [-]', [('wl', 'wavelength [nm]'), ('T', 'temperature [K]'), ('n', 'injected carriers concentration [1/cm]')]),
    'NR': ('Anisotropic complex refractive index tensor n<sub>R</sub> [-]. Tensor must have the form [ n<sub>00</sub>, n<sub>11</sub>, n<sub>22</sub>, n<sub>01</sub>, n<sub>10</sub> ]', [('wl', 'wavelength [nm]'), ('T', 'temperature [K]'), ('n', 'injected carriers concentration [1/cm]')]),
    'Nv': ('Effective density of states in the valance band N<sub>v</sub> [cm<sup>-3</sup>]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]'), ('point', 'point in the Brillouin zone [-]')]),
    'thermk': ('Thermal conductivity in in-plane (lateral) and cross-plane (vertical) direction k [W/(m K)]', [('T', 'temperature [K]'), ('h', 'layer thickness [µm]')]),
    'VB': ('Valance band level offset VB [eV]', [('T', 'temperature [K]'), ('e', 'lateral strain [-]'), ('hole', 'hole type (\'H\' or \'L\') [-]')]),
}

def materialHTMLHelp(property_name, font_size = None):
    prop_name, prop_attr = MATERIALS_PROPERTES.get(property_name, (None, None))
    res = ''
    if font_size != None: res += '<span style="font-size: %s">' % font_size
    if prop_name == None:
        res += "unknown property '%s'" % property_name
    else:
        res += prop_name + '<br>' + ', '.join(['<b>%s</b> - %s' % (n, v) for (n, v) in prop_attr])
    if font_size != None: res += '</span>'
    return res

class MaterialPropertyModel(QtCore.QAbstractTableModel):
    
    def __init__(self, materialsModel, material = None, parent=None, *args):
        QtCore.QAbstractListModel.__init__(self, parent, *args)
        self.materialsModel = materialsModel
        self.__material__ = material
    
    def rowCount(self, parent = QtCore.QModelIndex()):
        if not self.__material__ or parent.isValid(): return 0
        return len(self.__material__.properties)
    
    def columnCount(self, parent = QtCore.QModelIndex()): 
        return 3    # 3 if comment supported
    
    def get(self, col, row):
        n, v = self.__material__.properties[row]
        if col == 2:
            #prop_name, prop_attr = MATERIALS_PROPERTES[n]
            #return '<span style="font-size: 9pt">' + prop_name + '<br>' + ', '.join(['<b>%s</b> - %s' % (n, v) for (n, v) in prop_attr]) + '</span>'
            return materialHTMLHelp(n, '9pt')
        return n if col == 0 else v
    
    def data(self, index, role = QtCore.Qt.DisplayRole):
        if not index.isValid(): return None 
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole: 
            return self.get(index.column(), index.row())
#         if role == QtCore.Qt.ToolTipRole:
#             return '\n'.join([str(err) for err in self.info_by_row.get(index.row(), []) if err.has_connection('cols', index.column())])
#         if role == QtCore.Qt.DecorationRole: #QtCore.Qt.BackgroundColorRole:   #maybe TextColorRole?
#             max_level = -1
#             c = index.column()
#             for err in self.info_by_row.get(index.row(), []):
#                 if err.has_connection('cols', c, c == 0):   # c == 0 -> whole row massages has decoration only in first column
#                     if err.level > max_level: max_level = err.level
#             return info.infoLevelIcon(max_level)
        return None
    
    def set(self, col, row, value):
        n, v = self.__material__.properties[row]
        if col == 0:
            self.__material__.properties[row] = (value, v)
        elif col == 1:
            self.__material__.properties[row] = (n, value)
    
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        self.set(index.column(), index.row(), value)
        #self.fireChanged()
        self.dataChanged.emit(index, index)
        return True
    
    def flags(self, index):
        flags = super(MaterialPropertyModel, self).flags(index) | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled  

        if index.column() in [0, 1] and not self.materialsModel.isReadOnly(): flags |= QtCore.Qt.ItemIsEditable
        #flags |= QtCore.Qt.ItemIsDragEnabled
        #flags |= QtCore.Qt.ItemIsDropEnabled

        return flags
            
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'name'
            if col == 1: return 'value'
            if col == 2: return 'help'
        return None
    
    @property
    def material(self):
        return self.__material__
    
    @material.setter
    def material(self, material):
        self.layoutAboutToBeChanged.emit()
        self.__material__ = material
        self.layoutChanged.emit()
        
    def options_to_choose(self, index):
        """:return: list of available options to choose at given index or None"""
        if index.column() == 0: return MATERIALS_PROPERTES.keys()
        if index.column() == 1:
            if self.__material__.properties[index.row()][0] == 'condtype':
                return ['n', 'i', 'p', 'other']
        return None
        
        
        
class MaterialsModel(TableModel):
             
    class Material:
        def __init__(self, name, base = None, properties = [], comment = None):
            self.name = name
            self.base = base
            self.properties = properties    #TODO what with duplicate properties, should be supported?
            self.comment = comment
    
    def __init__(self, parent=None, info_cb = None, *args):
        TableModel.__init__(self, 'materials', parent, info_cb, *args)
        
    def setXMLElement(self, element):
        self.layoutAboutToBeChanged.emit()
        del self.entries[:]
        if isinstance(element, ElementTree.Element):
            for mat in element.iter("material"):
                self.entries.append(
                        MaterialsModel.Material(mat.attrib.get("name", ""), mat.attrib.get("base", None),  [ (prop.tag, prop.text) for prop in mat ])
                )
        self.layoutChanged.emit()
        self.fireChanged()
    
    # XML element that represents whole section
    def getXMLElement(self):
        res = ElementTree.Element(self.name)
        for e in self.entries:
            mat = ElementTree.SubElement(res, "material", { "name": e.name })
            mat.tail = '\n'
            if e.base: mat.attrib['base'] = e.base 
            if len(e.properties) > 0:
                mat.text = '\n  '
                prev = None
                for (n, v) in e.properties:
                    if prev is not None: prev.tail = '\n  '
                    p = ElementTree.SubElement(mat, n)
                    p.text = v
                    prev = p
                prev.tail = '\n'
        return res
    
    def get(self, col, row): 
        if col == 0: return self.entries[row].name
        if col == 1: return self.entries[row].base
        if col == 2: return self.entries[row].comment
        raise IndexError('column number for MaterialsModel should be 0, 1, or 2, but is %d' % col)
    
    def set(self, col, row, value):
        if col == 0: self.entries[row].name = value
        elif col == 1: self.entries[row].base = value
        elif col == 2: self.entries[row].comment = value
        else: raise IndexError('column number for MaterialsModel should be 0, 1, or 2, but is %d' % col)       
        
    def createDefaultEntry(self):
        return MaterialsModel.Material("name")
    
    # QAbstractListModel implementation
    
    def columnCount(self, parent = QtCore.QModelIndex()): 
        return 2    # 3 if comment supported
            
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'name'
            if col == 1: return 'base'
            if col == 2: return 'comment'
        return None
    
    def createInfo(self):
        res = super(MaterialsModel, self).createInfo()
        
        names = OrderedDict()
        for i, d in enumerate(self.entries):
            if not d.name:
                res.append(Info('Material name is required [row: %d]' % i, Info.ERROR, rows = [i], cols = [0]))
            else:
                names.setdefault(d.name, []).append(i)
            if not d.base:
                res.append(Info('Material base is required [row: %d]' % i, Info.ERROR, rows = [i], cols = [1]))
        for name, indexes in names.items():
            if len(indexes) > 1:
                res.append(Info('Duplicated material name "%s" [rows: %s]' % (name, ', '.join(map(str, indexes))),
                                Info.ERROR, rows = indexes, cols = [0]
                                )
                          )
        return res