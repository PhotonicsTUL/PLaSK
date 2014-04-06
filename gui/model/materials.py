# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui
from lxml import etree as ElementTree
from collections import OrderedDict

from .table import TableModel, TableModelEditMethods
from .info import Info
#from guis import DefinesEditor

MATERIALS_PROPERTES = OrderedDict((
    (u'A', (u'Monomolecular recombination coefficient <i>A</i>', '1/s', [(u'T', 'temperature [K]')])),
    (u'absb', (u'Absorption coefficient <i>α</i>', 'cm<sup>-1</sup>', [(u'wl', 'wavelength [nm]'), (u'T', 'temperature [K]')])),
    (u'ac', (u'Hydrostatic deformation potential for the conduction band <i>a<sub>c</sub></i>', 'eV', [(u'T', 'temperature [K]')])),
    (u'av', (u'Hydrostatic deformation potential for the valence band <i>a<sub>v</sub></i>', 'eV', [(u'T', 'temperature [K]')])),
    (u'B', (u'Radiative recombination coefficient <i>B</i>', 'm<sup>3</sup>/s', [(u'T', 'temperature [K]')])),
    (u'b', (u'Shear deformation potential <i>b</i>', 'eV', [(u'T', 'temperature [K]')])),
    (u'C', (u'Auger recombination coefficient <i>C</i>', 'm<sup>6</sup>/s', [(u'T', 'temperature [K]')])),
    (u'c11', (u'Elastic constant <i>c<sub>11</sub></i>', 'GPa', [(u'T', 'temperature [K]')])),
    (u'c12', (u'Elastic constant <i>c<sub>12</sub></i>', 'GPa', [(u'T', 'temperature [K]')])),
    (u'CB', (u'Conduction band level <i>CB</i>', 'eV', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]'), (u'point', 'point in the Brillouin zone [-]')])),
    (u'chi', (u'Electron affinity <i>χ</i>', 'eV', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]'), (u'point', 'point in the Brillouin zone [-]')])),
    (u'cond', (u'Electrical conductivity <i>σ</i> in-plane (lateral) and cross-plane (vertical) direction', 'S/m', [(u'T', 'temperature [K]')])),
    (u'condtype', (u'Electrical conductivity type. In semiconductors this indicates what type of carriers <i>Nf</i> refers to.', [])),
    (u'cp', (u'Specific heat at constant pressure', 'J/(kg K)', [(u'T', 'temperature [K]')])),
    (u'D', (u'Ambipolar diffusion coefficient <i>D</i>', 'm<sup>2</sup>/s', [(u'T', 'temperature [K]')])),
    (u'dens', (u'Density', 'kg/m<sup>3</sup>', [(u'T', 'temperature [K]')])),
    (u'Dso', (u'Split-off energy <i>D</i><sub>so</sub>', 'eV', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]')])),
    (u'EactA', (u'Acceptor ionization energy <i>E</i><sub>actA</sub>', 'eV', [(u'T', 'temperature [K]')])),
    (u'EactD', (u'Acceptor ionization energy <i>E</i><sub>actD</sub>', 'eV', [(u'T', 'temperature [K]')])),
    (u'Eg', (u'Energy gap <i>E<sub>g</sub></i>', 'eV', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]'), (u'point', 'point in the Brillouin')])),
    (u'eps', (u'Donor ionization energy <i>ε<sub>R</sub></i>', '-', [(u'T', 'temperature [K]')])),
    (u'lattC', (u'Lattice constant', 'Å', [(u'T', 'temperature [K]'), (u'x', 'lattice parameter [-]')])),
    (u'Me', (u'Electron effective mass <i>M<sub>e</sub></i> in in-plane (lateral) and cross-plane (vertical) direction', '<i>m</i><sub>0</sub>', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]'), (u'point', 'point in the irreducible Brillouin zone [-]')])),
    (u'Mh', (u'Hole effective mass <i>M<sub>h</sub></i> in in-plane (lateral) and cross-plane (vertical) direction', '<i>m</i><sub>0</sub>', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]')])),
    (u'Mhh', (u'Heavy hole effective mass <i>M<sub>hh</sub></i> in in-plane (lateral) and cross-plane (vertical) direction', '<i>m</i><sub>0</sub>', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]')])),
    (u'Mlh', (u'Light hole effective mass <i>M<sub>lh</sub></i> in in-plane (lateral) and cross-plane (vertical) direction', '<i>m</i><sub>0</sub>', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]')])),
    (u'Mso', (u'Split-off mass <i>M</i><sub>so</sub>', '<i>m</i><sub>0</sub>', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]')])),
    (u'Nc', (u'Effective density of states in the conduction band <i>Nc</i>', 'cm<sup>-3</sup>', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]'), (u'point', 'point in the Brillouin zone [-]')])),
    (u'Nf', (u'Free carrier concentration <i>N</i>', 'cm<sup>-3</sup>', [(u'T', 'temperature [K]')])),
    (u'Ni', (u'Intrinsic carrier concentration <i>N<sub>i</sub></i>', 'cm<sup>-3</sup>', [(u'T', 'temperature [K]')])),
    (u'Nr', (u'Complex refractive index <i>n<sub>R</sub></i>', '-', [(u'wl', 'wavelength [nm]'), (u'T', 'temperature [K]'), (u'n', 'injected carriers concentration [1/cm]')])),
    (u'nr', (u'Real refractive index <i>n<sub>R</sub></i>', '-', [(u'wl', 'wavelength [nm]'), (u'T', 'temperature [K]'), (u'n', 'injected carriers concentration [1/cm]')])),
    (u'NR', (u'Anisotropic complex refractive index tensor <i>n<sub>R</sub></i>. Tensor must have the form [<i>n</i><sub>00</sub>, <i>n</i><sub>11</sub>, <i>n</i><sub>22</sub>, <i>n</i><sub>01</sub>, <i>n</i><sub>10</sub>]', '-', [(u'wl', 'wavelength [nm]'), (u'T', 'temperature [K]'), (u'n', 'injected carriers concentration [1/cm]')])),
    (u'Nv', (u'Effective density of states in the valance band <i>N<sub>v</sub></i>', 'cm<sup>-3</sup>', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]'), (u'point', 'point in the Brillouin zone [-]')])),
    (u'thermk', (u'Thermal conductivity in in-plane (lateral) and cross-plane (vertical) direction <i>k</i>', 'W/(m K)', [(u'T', 'temperature [K]'), (u'h', 'layer thickness [µm]')])),
    (u'VB', (u'Valance band level offset <i>VB</i>', 'eV', [(u'T', 'temperature [K]'), (u'e', 'lateral strain [-]'), (u'hole', 'hole type (\'H\' or \'L\') [-]')])),
))

def materialHTMLHelp(property_name, with_unit=True, with_attr=False, font_size=None):
    prop_help, prop_unit, prop_attr = MATERIALS_PROPERTES.get(property_name, (None, None, None))
    res = ''
    if font_size is not None: res += '<span style="font-size: %s">' % font_size
    if prop_help is None:
        res += "unknown property '%s'" % property_name
    else:
        res += prop_help
        if with_unit and prop_unit is not None:
            res += ' [' + prop_unit + ']'
        if with_attr and prop_attr is not None and len(prop_attr) > 0:
            res += '<br>' + ', '.join(['<b><i>%s</i></b> - %s' % (n, v) for (n, v) in prop_attr])
    if font_size is not None: res += '</span>'
    return res

def materialUnit(property_name):
    return MATERIALS_PROPERTES.get(property_name, (None, '', None))[1]


class MaterialPropertyModel(QtCore.QAbstractTableModel, TableModelEditMethods):

    def __init__(self, materialsModel, material=None, parent=None, *args):
        QtCore.QAbstractListModel.__init__(self, parent, *args)
        TableModelEditMethods.__init__(self)
        self.materialsModel = materialsModel
        self.__material = material

    def rowCount(self, parent = QtCore.QModelIndex()):
        if not self.__material or parent.isValid(): return 0
        return len(self.__material.properties)

    def columnCount(self, parent = QtCore.QModelIndex()):
        return 4    # 5 if comment supported

    def get(self, col, row):
        n, v = self.__material.properties[row]
        if col == 2:
            return '<span style="font-size: 12pt;">%s</span>' % materialUnit(n)
        elif col == 3:
            #prop_name, prop_attr = MATERIALS_PROPERTES[n]
            #return '<span style="font-size: 9pt">' + prop_name + '<br>' + ', '.join(['<b>%s</b> - %s' % (n, v) for (n, v) in prop_attr]) + '</span>'
            return materialHTMLHelp(n, with_unit=False, with_attr=True, font_size='10pt')
        return n if col == 0 else v

    def data(self, index, role = QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return self.get(index.column(), index.row())
#         if role == QtCore.Qt.ToolTipRole:
#             return '\n'.join([str(err) for err in self.info_by_row.get(index.row(), []) if err.has_connection(u'cols', index.column())])
#         if role == QtCore.Qt.DecorationRole: #QtCore.Qt.BackgroundColorRole:   #maybe TextColorRole?
#             max_level = -1
#             c = index.column()
#             for err in self.info_by_row.get(index.row(), []):
#                 if err.has_connection(u'cols', c, c == 0):   # c == 0 -> whole row massages has decoration only in first column
#                     if err.level > max_level: max_level = err.level
#             return info.infoLevelIcon(max_level)
        if role == QtCore.Qt.BackgroundRole and index.column() >= 2:
            return QtGui.QBrush(QtGui.QPalette().color(QtGui.QPalette.Normal, QtGui.QPalette.Window))

    def set(self, col, row, value):
        n, v = self.__material.properties[row]
        if col == 0:
            self.__material.properties[row] = (value, v)
        elif col == 1:
            self.__material.properties[row] = (n, value)

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        self.set(index.column(), index.row(), value)
        #self.fire_changed()
        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        flags = super(MaterialPropertyModel, self).flags(index) | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        if index.column() in [0, 1] and not self.materialsModel.is_read_only(): flags |= QtCore.Qt.ItemIsEditable
        #flags |= QtCore.Qt.ItemIsDragEnabled
        #flags |= QtCore.Qt.ItemIsDropEnabled

        return flags

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            try:
                return ('Name', 'Value', 'Unit', 'Help')[col]
            except IndexError:
                return None

    @property
    def material(self):
        return self.__material

    @material.setter
    def material(self, material):
        self.layoutAboutToBeChanged.emit()
        self.__material = material
        self.layoutChanged.emit()

    def options_to_choose(self, index):
        """:return: list of available options to choose at given index or None"""
        if index.column() == 0: return MATERIALS_PROPERTES.keys()
        if index.column() == 1:
            if self.__material.properties[index.row()][0] == 'condtype':
                return ['n', 'i', 'p', 'other']
        return None

    @property
    def entries(self):
        return self.__material.properties

    def is_read_only(self):
        return self.material == None or self.materialsModel.is_read_only()

    def fire_changed(self):
        pass

    def create_default_entry(self):
        return "", ""


class MaterialsModel(TableModel):

    class Material: #(InfoSource)

        def __init__(self, name, base = None, properties = [], comment = None):
            self.name = name
            self.base = base
            self.properties = properties    #TODO what with duplicate properties, should be supported?
            self.comment = comment

        def add_to_XML(self, material_section_element):
            mat = ElementTree.SubElement(material_section_element, "material", { "name": self.name })
            if self.base: mat.attrib['base'] = self.base
            for (n, v) in self.properties:
                ElementTree.SubElement(mat, n).text = v

    def __init__(self, parent=None, info_cb = None, *args):
        super(MaterialsModel, self).__init__(u'materials', parent, info_cb, *args)

    def set_XML_element(self, element):
        self.layoutAboutToBeChanged.emit()
        del self.entries[:]
        if element is not None:
            for mat in element.iter("material"):
                self.entries.append(
                        MaterialsModel.Material(mat.attrib.get("name", ""), mat.attrib.get("base", None),  [ (prop.tag, prop.text) for prop in mat ])
                )
        self.layoutChanged.emit()
        self.fire_changed()

    # XML element that represents whole section
    def get_XML_element(self):
        res = ElementTree.Element(self.name)
        for e in self.entries:
            if e.comment: res.append(ElementTree.Comment(e.comment))
            e.add_to_XML(res)
        return res

    def get(self, col, row):
        if col == 0: return self.entries[row].name
        if col == 1: return self.entries[row].base
        if col == 2: return self.entries[row].comment
        raise IndexError(u'column number for MaterialsModel should be 0, 1, or 2, but is %d' % col)

    def set(self, col, row, value):
        if col == 0: self.entries[row].name = value
        elif col == 1: self.entries[row].base = value
        elif col == 2: self.entries[row].comment = value
        else: raise IndexError(u'column number for MaterialsModel should be 0, 1, or 2, but is %d' % col)

    def create_default_entry(self):
        return MaterialsModel.Material("name", "semiconductor")

    # QAbstractListModel implementation

    def columnCount(self, parent = QtCore.QModelIndex()):
        return 2    # 3 if comment supported

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if col == 0: return 'Name'
            if col == 1: return 'Base'
            if col == 2: return 'Comment'
        return None

    def create_info(self):
        res = super(MaterialsModel, self).create_info()

        names = OrderedDict()
        for i, d in enumerate(self.entries):
            if not d.name:
                res.append(Info(u'Material name is required [row: %d]' % i, Info.ERROR, rows = [i], cols = [0]))
            else:
                names.setdefault(d.name, []).append(i)
            if not d.base:
                res.append(Info(u'Material base is required [row: %d]' % i, Info.ERROR, rows = [i], cols = [1]))
        for name, indexes in names.items():
            if len(indexes) > 1:
                res.append(Info(u'Duplicated material name "%s" [rows: %s]' % (name, ', '.join(map(str, indexes))),
                                Info.ERROR, rows = indexes, cols = [0]
                                )
                          )
        return res
