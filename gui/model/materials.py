# -*- coding: utf-8 -*-
# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import re

from ..qt import QtCore, QtGui
from ..qt.QtCore import Qt

from lxml import etree as ElementTree
from collections import OrderedDict

from ..utils.widgets import DEFAULT_FONT
from .table import TableModel, TableModelEditMethods
from .info import Info
from ..utils.xml import OrderedTagReader, AttributeReader, require_no_attributes, require_no_children

try:
    import plask
except ImportError:
    plask = None
else:
    import plask.material


MATERIALS_PROPERTES = OrderedDict((
    ('A', (u'Monomolecular recombination coefficient <i>A</i>', u'1/s',
           [(u'T', u'temperature [K]')])),
    ('Ae', (u'Monomolecular recombination coefficient <i>A<sub>e</sub></i> for electrons', u'1/s',
           [(u'T', u'temperature [K]')])),
    ('Ah', (u'Monomolecular recombination coefficient <i>A<sub>h</sub></i> for holes', u'1/s',
           [(u'T', u'temperature [K]')])),
    ('absp', (u'Absorption coefficient <i>α</i>', u'cm<sup>-1</sup>',
              [(u'wl', u'wavelength [nm]'),
               (u'T', u'temperature [K]')])),
    ('ac', (u'Hydrostatic deformation potential for the conduction band <i>a<sub>c</sub></i>', u'eV',
            [(u'T', u'temperature [K]')])),
    ('av', (u'Hydrostatic deformation potential for the valence band <i>a<sub>v</sub></i>', u'eV',
            [(u'T', u'temperature [K]')])),
    ('B', (u'Radiative recombination coefficient <i>B</i>', u'cm<sup>3</sup>/s',
           [(u'T', u'temperature [K]')])),
    ('b', (u'Shear deformation potential <i>b</i>', u'eV',
           [(u'T', u'temperature [K]')])),
    ('C', (u'Auger recombination coefficient <i>C</i>', u'cm<sup>6</sup>/s',
           [(u'T', u'temperature [K]')])),
    ('Ce', (u'Auger recombination coefficient <i>C<sub>e</sub></i> for electrons', u'cm<sup>6</sup>/s',
           [(u'T', u'temperature [K]')])),
    ('Ch', (u'Auger recombination coefficient <i>C<sub>h</sub></i> for holes', u'cm<sup>6</sup>/s',
           [(u'T', u'temperature [K]')])),
    ('c11', (u'Elastic constant <i>c<sub>11</sub></i>', u'GPa',
             [(u'T', u'temperature [K]')])),
    ('c12', (u'Elastic constant <i>c<sub>12</sub></i>', u'GPa',
             [(u'T', u'temperature [K]')])),
    ('c13', (u'Elastic constant <i>c<sub>13</sub></i>', u'GPa',
             [(u'T', u'temperature [K]')])),
    ('c33', (u'Elastic constant <i>c<sub>33</sub></i>', u'GPa',
             [(u'T', u'temperature [K]')])),
    ('c44', (u'Elastic constant <i>c<sub>44</sub></i>', u'GPa',
             [(u'T', u'temperature [K]')])),
    ('CB', (u'Conduction band level <i>CB</i>', u'eV',
            [(u'T', u'temperature [K]'),
             (u'e', u'lateral strain [-]'),
             (u'point', u'point in the Brillouin zone [-]')])),
    ('chi', (u'Electron affinity <i>χ</i>', u'eV',
             [(u'T', u'temperature [K]'),
              (u'e', u'lateral strain [-]'),
              (u'point', u'point in the Brillouin zone [-]')])),
    ('cond', (u'Electrical conductivity <i>σ</i> lateral and vertical directions', u'S/m',
              [(u'T', u'temperature [K]')])),
    ('condtype', (u'Electrical conductivity type. '
                  u'In semiconductors this indicates what type of carriers <i>Nf</i> refers to.', u'-', [])),
    ('cp', (u'Specific heat at constant pressure', u'J/(kg K)',
            [(u'T', u'temperature [K]')])),
    ('D', (u'Ambipolar diffusion coefficient <i>D</i>', u'cm<sup>2</sup>/s',
           [(u'T', u'temperature [K]')])),
    ('dens', (u'Density', u'kg/m<sup>3</sup>',
              [(u'T', u'temperature [K]')])),
    ('Dso', (u'Split-off energy <i>D</i><sub>so</sub>', u'eV',
             [(u'T', u'temperature [K]'), (u'e', u'lateral strain [-]')])),
    ('e13', (u'Piezoelectric coeffcient <i>e<sub>13</sub></i>', u'C/m<sup>2</sup>',
             [(u'T', u'temperature [K]')])),
    ('e33', (u'Piezoelectric coeffcient <i>e<sub>33</sub></i>', u'C/m<sup>2</sup>',
             [(u'T', u'temperature [K]')])),
    ('EactA', (u'Acceptor ionization energy <i>E</i><sub>actA</sub>', u'eV',
               [(u'T', u'temperature [K]')])),
    ('EactD', (u'Donor ionization energy <i>E</i><sub>actD</sub>', u'eV',
               [(u'T', u'temperature [K]')])),
    ('Eg', (u'Energy gap <i>E<sub>g</sub></i>', u'eV',
            [(u'T', u'temperature [K]'),
             (u'e', u'lateral strain [-]'),
             (u'point', u'point in the Brillouin')])),
    ('eps', (u'Dielectric constant <i>ε<sub>R</sub></i>', u'-',
             [(u'T', u'temperature [K]')])),
    ('lattC', (u'Lattice constant', u'Å',
               [(u'T', u'temperature [K]'),
                (u'x', u'lattice parameter [-]')])),
    ('Me', (u'Electron effective mass <i>M<sub>e</sub></i> in lateral '
            u'and verical directions', u'<i>m</i><sub>0</sub>',
            [(u'T', u'temperature [K]'),
             (u'e', u'lateral strain [-]'),
             (u'point', u'point in the irreducible Brillouin zone [-]')])),
    ('Mh', (u'Hole effective mass <i>M<sub>h</sub></i> in lateral '
            u'and verical directions', u'<i>m</i><sub>0</sub>',
            [(u'T', u'temperature [K]'),
             (u'e', u'lateral strain [-]')])),
    ('Mhh', (u'Heavy hole effective mass <i>M<sub>hh</sub></i> in lateral '
             u'and verical directions', u'<i>m</i><sub>0</sub>',
             [(u'T', u'temperature [K]'), (u'e', u'lateral strain [-]')])),
    ('Mlh', (u'Light hole effective mass <i>M<sub>lh</sub></i> in lateral '
             u'and verical directions', u'<i>m</i><sub>0</sub>',
             [(u'T', u'temperature [K]'), (u'e', u'lateral strain [-]')])),
    ('mob', (u'Majority carriers mobility <i>µ</i> lateral and vertical directions', u'cm<sup>2</sup>/(V s)',
              [(u'T', u'temperature [K]')])),
    ('mobe', (u'Electron mobility <i>µ<sub>e</sub></i> lateral and vertical directions', u'cm<sup>2</sup>/(V s)',
              [(u'T', u'temperature [K]')])),
    ('mobh', (u'Hole mobility <i>µ<sub>h</sub></i> lateral and vertical directions', u'cm<sup>2</sup>/(V s)',
              [(u'T', u'temperature [K]')])),
    ('Mso', (u'Split-off mass <i>M</i><sub>so</sub>', u'<i>m</i><sub>0</sub>',
             [(u'T', u'temperature [K]'),
              (u'e', u'lateral strain [-]')])),
    ('Nc', (u'Effective density of states in the conduction band <i>N<sub>c</sub></i>', u'cm<sup>-3</sup>',
            [(u'T', u'temperature [K]'),
             (u'e', u'lateral strain [-]'),
             (u'point', u'point in the Brillouin zone [-]')])),
    ('Nf', (u'Free carrier concentration <i>N</i>', u'cm<sup>-3</sup>',
            [(u'T', u'temperature [K]')])),
    ('Ni', (u'Intrinsic carrier concentration <i>N<sub>i</sub></i>', u'cm<sup>-3</sup>',
            [(u'T', u'temperature [K]')])),
    ('Nr', (u'Complex refractive index <i>n<sub>R</sub></i>', u'-',
            [(u'wl', u'wavelength [nm]'),
             (u'T', u'temperature [K]'),
             (u'n', u'injected carriers concentration [1/cm]')])),
    ('nr', (u'Real refractive index <i>n<sub>R</sub></i>', u'-',
            [(u'wl', u'wavelength [nm]'), (u'T', u'temperature [K]'),
             (u'n', u'injected carriers concentration [1/cm]')])),
    ('NR', (u'Anisotropic complex refractive index tensor <i>n<sub>R</sub></i>. '
            u'Tensor must have the form [<i>n</i><sub>00</sub>, <i>n</i><sub>11</sub>, '
            u'<i>n</i><sub>22</sub>, <i>n</i><sub>01</sub>, <i>n</i><sub>10</sub>]', u'-',
            [(u'wl', u'wavelength [nm]'), (u'T', u'temperature [K]'),
             (u'n', u'injected carriers concentration [1/cm]')])),
    ('Nv', (u'Effective density of states in the valance band <i>N<sub>v</sub></i>', u'cm<sup>-3</sup>',
            [(u'T', u'temperature [K]'), (u'e', u'lateral strain [-]'),
             (u'point', u'point in the Brillouin zone [-]')])),
    ('Psp', (u'Spontaneous polarization <i>P<sub>sp</sub></i>', u'C/m<sup>2</sup>',
              [(u'T', u'temperature [K]')])),
    ('thermk', (u'Thermal conductivity in lateral and vertical directions <i>k</i>', u'W/(m K)',
                [(u'T', u'temperature [K]'),
                 (u'h', u'layer thickness [µm]')])),
    ('VB', (u'Valance band level offset <i>VB</i>', u'eV',
            [(u'T', u'temperature [K]'), (u'e', u'lateral strain [-]'),
             (u'hole', u'hole type (\'H\' or \'L\') [-]')])),
))

ELEMENT_GROUPS = (('Al', 'Ga', 'In'), ('N', 'P', 'As', 'Sb', 'Bi'))


elements_re = re.compile("[A-Z][a-z]*")


def parse_material_components(material):
    """ Parse info on materials.
        :return: name, label, groups, doping
    """
    if plask:
        try:
            simple = plask.material.db.is_simple(str(material))
        except ValueError:
            simple = True
        except RuntimeError:
            simple = True
    else:
        simple = True
    if ':' in material:
        name, doping = material.split(':')
    else:
        name = material
        doping = None
    if '_' in name:
        name, label = name.split('_')
    else:
        label = ''
    if not simple:
        elements = elements_re.findall(name)
        groups = [[e for e in elements if e in g] for g in ELEMENT_GROUPS]
    else:
        groups = []
    return name, label, groups, doping


def material_html_help(property_name, with_unit=True, with_attr=False, font_size=None):
    prop_help, prop_unit, prop_attr = MATERIALS_PROPERTES.get(property_name, (None, None, None))
    res = ''
    if font_size is not None: res += '<span style="font-size: %s">' % font_size
    if prop_help is None:
        res += "unknown property"
    else:
        res += prop_help
        if with_unit and prop_unit is not None:
            res += ' [' + prop_unit + ']'
        if with_attr and prop_attr is not None and len(prop_attr) > 0:
            res += '<br>' + ', '.join(['<b><i>%s</i></b> - %s' % (n, v) for (n, v) in prop_attr])
    if font_size is not None: res += '</span>'
    return res


def material_unit(property_name):
    return MATERIALS_PROPERTES.get(property_name, (None, '', None))[1]


class MaterialsModel(TableModel):

    class External(object):

        def __init__(self, materials_model, what, name='', comment=None):
            self.materials_model = materials_model
            self.what = what
            self.name = name
            self.comment = comment

        def add_to_xml(self, material_section_element):
            ElementTree.SubElement(material_section_element, self.what, {"name": self.name})

        @property
        def base(self):
            return {'library': 'Binary Library', 'module': 'Python Module'}[self.what]

    class Material(TableModelEditMethods, QtCore.QAbstractTableModel): #(InfoSource)

        def __init__(self, materials_model, name, base=None, properties=None, comment=None, parent=None, *args):
            QtCore.QAbstractTableModel.__init__(self, parent, *args)
            TableModelEditMethods.__init__(self)
            self.materials_model = materials_model
            if properties is None: properties = []
            self.name = name
            self.base = base
            self.properties = properties
            self.comment = comment

        def add_to_xml(self, material_section_element):
            mat = ElementTree.SubElement(material_section_element, "material", {"name": self.name})
            if self.base: mat.attrib['base'] = self.base
            for (n, v) in self.properties:
                ElementTree.SubElement(mat, n).text = v

        def rowCount(self, parent=QtCore.QModelIndex()):
            if parent.isValid(): return 0
            return len(self.properties)

        def columnCount(self, parent=QtCore.QModelIndex()):
            return 4    # 5 if comment supported

        def get(self, col, row):
            n, v = self.properties[row]
            if col == 2:
                return material_unit(n)
            elif col == 3:
                return material_html_help(n, with_unit=False, with_attr=True)
            return n if col == 0 else v

        get_raw = get

        def data(self, index, role=Qt.DisplayRole):
            if not index.isValid(): return None
            if role == Qt.DisplayRole or role == Qt.EditRole:
                return self.get(index.column(), index.row())
    #         if role == Qt.ToolTipRole:
    #             return '\n'.join(str(err) for err in self.info_by_row.get(index.row(), [])
    #                              if err.has_connection(u'cols', index.column())s)
    #         if role == Qt.DecorationRole: #Qt.BackgroundColorRole:   #maybe TextColorRole?
    #             max_level = -1
    #             c = index.column()
    #             for err in self.info_by_row.get(index.row(), []):
    #                 if err.has_connection(u'cols', c, c == 0):
    #                     if err.level > max_level: max_level = err.level
    #             return info.infoLevelIcon(max_level)
            if role == Qt.BackgroundRole and index.column() >= 2:
                return QtGui.QBrush(QtGui.QPalette().color(QtGui.QPalette.Normal, QtGui.QPalette.Window))

        def set(self, col, row, value):
            n, v = self.properties[row]
            if col == 0:
                self.properties[row] = (value, v)
            elif col == 1:
                self.properties[row] = (n, value)

        def flags(self, index):
            flags = super(MaterialsModel.Material, self).flags(index) | Qt.ItemIsSelectable | Qt.ItemIsEnabled

            if index.column() in [0, 1] and not self.materials_model.is_read_only(): flags |= Qt.ItemIsEditable
            #flags |= Qt.ItemIsDragEnabled
            #flags |= Qt.ItemIsDropEnabled

            return flags

        def headerData(self, col, orientation, role):
            if orientation == Qt.Horizontal and role == Qt.DisplayRole:
                try:
                    return ('Name', 'Value', 'Unit', 'Help')[col]
                except IndexError:
                    return None

        def options_to_choose(self, index):
            """:return: list of available options to choose at given index or None"""
            if index.column() == 0: return MATERIALS_PROPERTES.keys()
            if index.column() == 1:
                if self.properties[index.row()][0] == 'condtype':
                    return ['n', 'i', 'p', 'other']
            return None

        @property
        def entries(self):
            return self.properties

        def is_read_only(self):
            return self.materials_model.is_read_only()

        def fire_changed(self):
            self.materials_model.fire_changed()

        def create_default_entry(self):
            return "", ""

        @property
        def undo_stack(self):
            return self.materials_model.undo_stack

    def __init__(self, parent=None, info_cb=None, *args):
        super(MaterialsModel, self).__init__(u'materials', parent, info_cb, *args)

    def set_xml_element(self, element, undoable=True):
        new_entries = []
        for mat in element:
            if mat.tag == 'material':
                with AttributeReader(mat) as mat_attrib:
                    properties = []
                    for prop in mat:
                        require_no_children(prop)
                        require_no_attributes(prop)
                        properties.append((prop.tag, prop.text))
                    base = mat_attrib.get('base', None)
                    if base is None: base = mat_attrib.get('kind')  # for old files
                    new_entries.append(MaterialsModel.Material(self, mat_attrib.get('name', ''), base, properties))
            elif mat.tag in ('library', 'module'):
                new_entries.append(MaterialsModel.External(self, mat.tag, mat.attrib.get('name', '')))
        self._set_entries(new_entries, undoable)

    # XML element that represents whole section
    def get_xml_element(self):
        res = ElementTree.Element(self.name)
        for e in self.entries:
            if e.comment: res.append(ElementTree.Comment(e.comment))
            e.add_to_xml(res)
        return res

    def data(self, index, role=Qt.DisplayRole):
        if isinstance(self.entries[index.row()], MaterialsModel.External):
            if index.column() == 1:
                if role == Qt.FontRole:
                    font = QtGui.QFont()
                    font.setItalic(True)
                    return font
            elif index.column() == 0 and role == Qt.DecorationRole:
                return QtGui.QIcon.fromTheme(
                    {'library': 'emblem-system', 'module': 'application-x-executable'}[self.entries[index.row()].what])
        return super(MaterialsModel, self).data(index, role)

    def flags(self, index):
        flags = super(MaterialsModel, self).flags(index)
        if index.column() == 1 and isinstance(self.entries[index.row()], MaterialsModel.External):
            flags &= ~Qt.ItemIsEditable & ~Qt.ItemIsEnabled
        return flags

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
        return MaterialsModel.Material(self, "name", "semiconductor")

    # QAbstractListModel implementation

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2    # 3 if comment supported

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if col == 0: return 'Name'
            if col == 1: return 'Base'
            if col == 2: return 'Comment'
        return None

    def create_info(self):
        res = super(MaterialsModel, self).create_info()

        names = OrderedDict()
        for i, d in enumerate(self.entries):
            if isinstance(d, MaterialsModel.Material):
                if not d.name:
                    res.append(Info(u'Material name is required [row: {}]'.format(i+1), Info.ERROR, rows=[i], cols=[0]))
                else:
                    names.setdefault(d.name, []).append(i)
                if not d.base:
                    res.append(Info(u'Material base is required [row: {}]'.format(i+1), Info.ERROR, rows=[i], cols=[1]))
                elif plask and d.base not in (e.name for e in self.entries[:i]) and '{' not in d.base:
                    try:
                        mat = str(d.base)
                        if ':'  in mat and '=' not in mat:
                            mat += '=0'
                        plask.material.db.get(mat)
                    except (ValueError, RuntimeError) as err:
                        res.append(
                            Info(u"Material base '{1}' is not a proper material ({2}) [row: {0}]"
                                 .format(i+1, d.base, err), Info.ERROR, rows=[i], cols=[1]))

        for name, rows in names.items():
            if len(rows) > 1:
                res.append(
                    Info(u'Duplicated material name "{}" [rows: {}]'.format(name, ', '.join(str(i+1) for i in rows)),
                         Info.ERROR, rows=rows, cols=[0]))
        return res
