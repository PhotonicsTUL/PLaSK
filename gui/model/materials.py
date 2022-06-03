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
import sys
import os
import re
from copy import copy

from lxml import etree
import itertools

from importlib import import_module
try: from importlib import reload as reload_module
except ImportError: reload_module = reload

from ..qt.QtCore import *
from ..qt.QtGui import *
from ..qt.QtWidgets import *
from .table import TableModel, TableModelEditMethods
from .info import Info
from ..utils.xml import AttributeReader, require_no_attributes, require_no_children, OrderedTagReader, \
                        get_text_unindent, make_indented_text

try:
    import plask
except ImportError:
    plask = None
else:
    import plask.material


BASE_MATERIALS = ['dielectric', 'liquid_crystal', 'metal', 'semiconductor']


MATERIALS_PROPERTES = {
    'A': (u'Monomolecular recombination coefficient <i>A</i>', u'1/s',
           [(u'T', u'temperature', u'K')]),
    'absp': (u'Absorption coefficient <i>α</i>', u'cm<sup>-1</sup>',
              [(u'lam', u'wavelength', u'nm'),
               (u'T', u'temperature', u'K')]),
    'ac': (u'Hydrostatic deformation potential for the conduction band <i>a</i><sub><i>c</i></sub>', u'eV',
            [(u'T', u'temperature', u'K')]),
    'av': (u'Hydrostatic deformation potential for the valence band <i>a</i><sub><i>v</i></sub>', u'eV',
            [(u'T', u'temperature', u'K')]),
    'B': (u'Radiative recombination coefficient <i>B</i>', u'cm<sup>3</sup>/s',
           [(u'T', u'temperature', u'K')]),
    'b': (u'Shear deformation potential <i>b</i>', u'eV',
           [(u'T', u'temperature', u'K')]),
    'C': (u'Auger recombination coefficient <i>C</i>', u'cm<sup>6</sup>/s',
           [(u'T', u'temperature', u'K')]),
    'Ce': (u'Auger recombination coefficient <i>C</i><sub><i>e</i></sub> for electrons', u'cm<sup>6</sup>/s',
           [(u'T', u'temperature', u'K')]),
    'Ch': (u'Auger recombination coefficient <i>C</i><sub><i>h</i></sub> for holes', u'cm<sup>6</sup>/s',
           [(u'T', u'temperature', u'K')]),
    'c11': (u'Elastic constant <i>c<sub>11</sub></i>', u'GPa',
             [(u'T', u'temperature', u'K')]),
    'c12': (u'Elastic constant <i>c<sub>12</sub></i>', u'GPa',
             [(u'T', u'temperature', u'K')]),
    'c13': (u'Elastic constant <i>c<sub>13</sub></i>', u'GPa',
             [(u'T', u'temperature', u'K')]),
    'c33': (u'Elastic constant <i>c<sub>33</sub></i>', u'GPa',
             [(u'T', u'temperature', u'K')]),
    'c44': (u'Elastic constant <i>c<sub>44</sub></i>', u'GPa',
             [(u'T', u'temperature', u'K')]),
    'CB': (u'Conduction band level <i>CB</i>', u'eV',
            [(u'T', u'temperature', u'K'),
             (u'e', u'lateral strain', '-'),
             (u'point', u'point in the Brillouin zone', '-')]),
    'chi': (u'Electron affinity <i>χ</i>', u'eV',
             [(u'T', u'temperature', u'K'),
              (u'e', u'lateral strain', '-'),
              (u'point', u'point in the Brillouin zone', '-')]),
    'cond': (u'Electrical conductivity <i>σ</i> in lateral and vertical directions', u'S/m',
              [(u'T', u'temperature', u'K')]),
    'condtype': (u'Electrical conductivity type. '
                  u'In semiconductors this indicates what type of carriers <i>Nf</i> refers to.', u'-', []),
    'cp': (u'Specific heat at constant pressure', u'J/(kg K)',
            [(u'T', u'temperature', u'K')]),
    'D': (u'Ambipolar diffusion coefficient <i>D</i>', u'cm<sup>2</sup>/s',
           [(u'T', u'temperature', u'K')]),
    'dens': (u'Density', u'kg/m<sup>3</sup>',
              [(u'T', u'temperature', u'K')]),
    'Dso': (u'Split-off energy <i>D</i><sub>so</sub>', u'eV',
             [(u'T', u'temperature', u'K'), (u'e', u'lateral strain', '-')]),
    'e13': (u'Piezoelectric coeffcient <i>e<sub>13</sub></i>', u'C/m<sup>2</sup>',
             [(u'T', u'temperature', u'K')]),
    'e15': (u'Piezoelectric coeffcient <i>e<sub>15</sub></i>', u'C/m<sup>2</sup>',
             [(u'T', u'temperature', u'K')]),
    'e33': (u'Piezoelectric coeffcient <i>e<sub>33</sub></i>', u'C/m<sup>2</sup>',
             [(u'T', u'temperature', u'K')]),
    'EactA': (u'Acceptor ionization energy <i>E</i><sub>actA</sub>', u'eV',
               [(u'T', u'temperature', u'K')]),
    'EactD': (u'Donor ionization energy <i>E</i><sub>actD</sub>', u'eV',
               [(u'T', u'temperature', u'K')]),
    'Eg': (u'Energy band gap <i>E</i><sub><i>g</i></sub>', u'eV',
            [(u'T', u'temperature', u'K'),
             (u'e', u'lateral strain', '-'),
             (u'point', u'point in the Brillouin zone', '-')]),
    'eps': (u'Dielectric constant <i>ε<sub>R</sub></i>', u'-',
             [(u'T', u'temperature', u'K')]),
    'lattC': (u'Lattice constant', u'Å',
               [(u'T', u'temperature', u'K'),
                (u'x', u'lattice parameter', '-')]),
    'Me': (u'Electron effective mass <i>M</i><sub><i>e</i></sub> in lateral '
            u'and verical directions', u'<i>m</i><sub>0</sub>',
            [(u'T', u'temperature', u'K'),
             (u'e', u'lateral strain', '-'),
             (u'point', u'point in the irreducible Brillouin zone', '-')]),
    'Mh': (u'Hole effective mass <i>M</i><sub><i>h</i></sub> in lateral '
            u'and verical directions', u'<i>m</i><sub>0</sub>',
            [(u'T', u'temperature', u'K'),
             (u'e', u'lateral strain', '-')]),
    'Mhh': (u'Heavy hole effective mass <i>M<sub>hh</sub></i> in lateral '
             u'and verical directions', u'<i>m</i><sub>0</sub>',
             [(u'T', u'temperature', u'K'), (u'e', u'lateral strain', '-')]),
    'Mlh': (u'Light hole effective mass <i>M<sub>lh</sub></i> in lateral '
             u'and verical directions', u'<i>m</i><sub>0</sub>',
             [(u'T', u'temperature', u'K'), (u'e', u'lateral strain', '-')]),
    'mob': (u'Majority carriers mobility <i>µ</i> in lateral and vertical directions', u'cm<sup>2</sup>/(V s)',
              [(u'T', u'temperature', u'K')]),
    'mobe': (u'Electron mobility <i>µ</i><sub><i>e</i></sub> in lateral and vertical directions', u'cm<sup>2</sup>/(V s)',
              [(u'T', u'temperature', u'K')]),
    'mobh': (u'Hole mobility <i>µ</i><sub><i>h</i></sub> in lateral and vertical directions', u'cm<sup>2</sup>/(V s)',
              [(u'T', u'temperature', u'K')]),
    'Mso': (u'Split-off mass <i>M</i><sub>so</sub>', u'<i>m</i><sub>0</sub>',
             [(u'T', u'temperature', u'K'),
              (u'e', u'lateral strain', '-')]),
    'Na': (u'Acceptor concentration <i>N</i><sub><i>a</i></sub>', u'cm<sup>-3</sup>', []),
    'Nd': (u'Donor concentration <i>N</i><sub><i>d</i></sub>', u'cm<sup>-3</sup>', []),
    'Nf': (u'Free carrier concentration <i>N</i>', u'cm<sup>-3</sup>',
            [(u'T', u'temperature', u'K')]),
    'Ni': (u'Intrinsic carrier concentration <i>N</i><sub><i>i</i></sub>', u'cm<sup>-3</sup>',
            [(u'T', u'temperature', u'K')]),
    'Nr': (u'Complex refractive index <i>n</i><sub><i>R</i></sub>', u'-',
            [(u'lam', u'wavelength', u'nm'),
             (u'T', u'temperature', u'K'),
             (u'n', u'injected carriers concentration', 'cm<sup>-1</sup>')]),
    'nr': (u'Real refractive index <i>n</i><sub><i>R</i></sub>', u'-',
            [(u'lam', u'wavelength', u'nm'), (u'T', u'temperature', u'K'),
             (u'n', u'injected carriers concentration', 'cm<sup>-1</sup>')]),
    'NR': (u'Anisotropic complex refractive index tensor <i>n</i><sub><i>R</i></sub>.<br/>'
            u'(mind that some solvers use Nr instead; '
            u'tensor must have the form [<i>n</i><sub>00</sub>, <i>n</i><sub>11</sub>, '
            u'<i>n</i><sub>22</sub>, <i>n</i><sub>01</sub>, <i>n</i><sub>10</sub>])', u'-',
            [(u'lam', u'wavelength', u'nm'), (u'T', u'temperature', u'K'),
             (u'n', u'injected carriers concentration', 'cm<sup>-1</sup>')]),
    'Psp': (u'Spontaneous polarization <i>P<sub>sp</sub></i>', u'C/m<sup>2</sup>',
              [(u'T', u'temperature', u'K')]),
    'taue': (u'Monomolecular electrons lifetime <i>τ<sub>e</sub></i>', u'ns',
           [(u'T', u'temperature', u'K')]),
    'tauh': (u'Monomolecular holes lifetime <i>A</i><sub><i>h</i></sub>', u'ns',
           [(u'T', u'temperature', u'K')]),
    'thermk': (u'Thermal conductivity in lateral and vertical directions <i>k</i>', u'W/(m K)',
                [(u'T', u'temperature', u'K'),
                 (u'h', u'layer thickness', u'µm')]),
    'VB': (u'Valance band level offset <i>VB</i>', u'eV',
            [(u'T', u'temperature', u'K'), (u'e', u'lateral strain', '-'),
             (u'point', u'point in the Brillouin zone', '-'),
             (u'hole', u'hole type (\'H\' or \'L\')', '-')]),
    'y1': (u'Luttinger parameter <i>γ</i><sub>1</sub>', '-', []),
    'y2': (u'Luttinger parameter <i>γ</i><sub>2</sub>', '-', []),
    'y3': (u'Luttinger parameter <i>γ</i><sub>3</sub>', '-', []),
}

ELEMENT_GROUPS = (('Al', 'Ga', 'In'), ('N', 'P', 'As', 'Sb', 'Bi'))


elements_re = re.compile(r"([A-Z][a-z]*)(?:\((\d*\.?\d*)\))?")


if plask is not None:
    default_materialdb = copy(plask.material.db)
else:
    default_materialdb = []


def parse_material_components(material, alloy=None):
    """ Parse info on materials.
        :return: name, label, groups, doping
    """
    material = str(material)
    if alloy is None:
        if plask:
            try:
                mat = plask.material.db.get(material)
            except (ValueError, RuntimeError):
                try:
                    alloy = plask.material.db.is_alloy(material)
                except (ValueError, RuntimeError):
                    alloy = False
            else:
                alloy = mat.alloy
        else:
            alloy = False
    if ':' in material:
        name, doping = material.split(':', 1)
    else:
        name = material
        doping = None
    if '_' in name:
        name, label = name.split('_', 1)
    else:
        label = ''
    if alloy:
        elements = elements_re.findall(name)
        groups = [[e for e in elements if e[0] in g] for g in ELEMENT_GROUPS]
    else:
        groups = []
    return name, label, groups, doping


def material_html_help(property_name, with_unit=True, with_attr=False, font_size=None):
    prop_help, prop_unit, prop_attr = MATERIALS_PROPERTES.get(property_name, (None, None, None))
    res = u''
    if font_size is not None: res += u'<span style="font-size: %s">' % font_size
    if prop_help is None:
        res += u"unknown property"
    else:
        res += prop_help
        if with_unit and prop_unit is not None:
            res += u' [' + prop_unit + u']'
        if with_attr and prop_attr is not None and len(prop_attr) > 0:
            res += u'<br>' + u', '.join(u'<b><i>{0}</i></b> - {1} [{2}]'.format(*attr) for attr in prop_attr)
    if font_size is not None: res += '</span>'
    return res


def material_unit(property_name):
    return MATERIALS_PROPERTES.get(property_name, (None, '', None))[1]


if plask is not None:

    class HandleMaterialsModule:

        def __init__(self, document):
            if document is not None and document.filename is not None:
                self.dirname = os.path.dirname(document.filename)
            else:
                self.dirname = None

        def __enter__(self):
            sys.path.insert(0, '.')
            if self.dirname is not None:
                sys.path.insert(0, self.dirname)
            return self

        def __exit__(self, type=None, value=None, traceback=None):
            if self.dirname is not None:
                del sys.path[0]
            del sys.path[0]

else:

    class HandleMaterialsModule:

        def __init__(self, names=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, type=None, value=None, traceback=None):
            pass


class MaterialsModel(TableModel):

    if plask is not None:
        class _HandleMaterialsModule(HandleMaterialsModule):

            class Register:
                def __init__(self, handler, fun):
                    self.handler = handler
                    self.fun = fun
                def __call__(self, name, cls, base):
                    try: self.handler.names.remove(name)
                    except ValueError: pass
                    self.handler.names.append(name)
                    return self.fun(name, cls, base)

            def __init__(self, document):
                super().__init__(document)
                self.names = []
                self._register_simple = plask._material._register_material_simple
                self._register_alloy = plask._material._register_material_alloy

            def __enter__(self):
                super().__enter__()
                plask._material._register_material_simple = self.Register(self, self._register_simple)
                plask._material._register_material_alloy = self.Register(self, self._register_alloy)
                return self

            def __exit__(self, type=None, value=None, traceback=None):
                super().__exit__(type, value, traceback)
                plask._material._register_material_simple = self._register_simple
                plask._material._register_material_alloy = self._register_alloy

    else:
        class _HandleMaterialsModule(HandleMaterialsModule):
            pass

    class External(QAbstractTableModel):

        def __init__(self, materials_model, what, name='', comments=None):
            super().__init__()
            self.materials_model = materials_model
            self.what = what
            self.name = name
            self.comments = [] if comments is None else comments
            self.cache = None

        def make_xml_element(self):
            return etree.Element(self.what, {"name": self.name})

        @property
        def base(self):
            return {'library': 'Binary Library', 'module': 'Python Module'}[self.what]

        def columnCount(self, parent=QModelIndex()):
            return 1

        def headerData(self, col, orientation, role):
            if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole and col == 0:
                return 'Materials in the ' + self.what
            return None

        def rowCount(self, parent=QModelIndex()):
            if self.cache is None: return 0
            return len(self.cache)

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid() or self.cache is None: return None
            if role == Qt.ItemDataRole.DisplayRole:
                return self.cache[index.row()]

    class Material(TableModelEditMethods, QAbstractTableModel): #(InfoSource)

        class Property:
            def __init__(self, name=None, value=None, comments=None):
                self.name = name
                self.value = value
                self.comments = [] if comments is None else comments

            def add_to_xml(self, material_element):
                if not self.name: return
                for c in self.comments:
                    material_element.append(etree.Comment(c))
                el = etree.SubElement(material_element, self.name)
                if self.value:
                    el.text = make_indented_text(self.value, 2)

        def __init__(self, materials_model, name, base=None, properties=None, alloy=False, comments=None,
                     endcomments=None, parent=None, *args):
            QAbstractTableModel.__init__(self, parent, *args)
            TableModelEditMethods.__init__(self)
            self.materials_model = materials_model
            if properties is None: properties = []
            self.name = name
            self.base = base
            self.properties = properties
            self.comments = [] if comments is None else comments
            self.endcomments = [] if endcomments is None else endcomments
            self.alloy = alloy

        def make_xml_element(self):
            mat = etree.Element("material", {"name": self.name})
            if self.base: mat.attrib['base'] = self.base
            if self.alloy: mat.attrib['alloy'] = 'yes'
            for p in self.properties:
                p.add_to_xml(mat)
            for c in self.endcomments:
                mat.append(etree.Comment(c))
            return mat

        def rowCount(self, parent=QModelIndex()):
            if parent.isValid(): return 0
            return len(self.properties)

        def columnCount(self, parent=QModelIndex()):
            return 4    # 5 if comment supported

        def get(self, col, row):
            p = self.properties[row]
            if col == 2:
                return material_unit(p.name)
            elif col == 3:
                return material_html_help(p.name, with_unit=False, with_attr=True)
            return p.name if col == 0 else p.value

        get_raw = get

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid(): return None
            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                return self.get(index.column(), index.row())
    #         if role == Qt.ItemDataRole.ToolTipRole:
    #             return '\n'.join(str(err) for err in self.info_by_row.get(index.row(), [])
    #                              if err.has_connection(u'cols', index.column())s)
    #         if role == Qt.ItemDataRole.DecorationRole: #Qt.ItemDataRole.BackgroundColorRole:   #maybe TextColorRole?
    #             max_level = -1
    #             c = index.column()
    #             for err in self.info_by_row.get(index.row(), []):
    #                 if err.has_connection(u'cols', c, c == 0):
    #                     if err.level > max_level: max_level = err.level
    #             return info.info_level_icon(max_level)
            if role == Qt.ItemDataRole.BackgroundRole and index.column() >= 2:
                return QBrush(QPalette().color(QPalette.ColorGroup.Normal, QPalette.ColorRole.Window))

        def set(self, col, row, value):
            p = self.properties[row]
            if col == 0:
                p.name = value
            elif col == 1:
                p.value = value

        def flags(self, index):
            flags = super().flags(index) | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

            if index.column() in [0, 1] and not self.materials_model.is_read_only(): flags |= Qt.ItemFlag.ItemIsEditable
            #flags |= Qt.ItemFlag.ItemIsDragEnabled
            #flags |= Qt.ItemFlag.ItemIsDropEnabled

            return flags

        def headerData(self, col, orientation, role):
            if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
                try:
                    return ('Name', 'Value', 'Unit', 'Help')[col]
                except IndexError:
                    return None

        def options_to_choose(self, index):
            """:return: list of available options to choose at given index or None"""
            if index.column() == 0: return MATERIALS_PROPERTES.keys()
            if index.column() == 1:
                if self.properties[index.row()].name == 'condtype':
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
            return MaterialsModel.Material.Property()

        @property
        def undo_stack(self):
            return self.materials_model.undo_stack

    def __init__(self, parent=None, info_cb=None, *args):
        super().__init__(u'materials', parent, info_cb, *args)
        self.document = None

    def load_xml_element(self, element, undoable=True):
        new_entries = []
        with OrderedTagReader(element) as materials:
            for mat in materials:
                if mat.tag == 'material':
                    with OrderedTagReader(mat) as props:
                        with AttributeReader(mat) as mat_attrib:
                            properties = []
                            for prop in props:
                                require_no_children(prop)
                                with AttributeReader(prop) as _:
                                    value = get_text_unindent(prop)
                                    properties.append(MaterialsModel.Material.Property(prop.tag, value, prop.comments))
                            base = mat_attrib.get('base', None)
                            if base is None: base = mat_attrib.get('kind')  # for old files
                            alloy = mat_attrib.get('alloy', '').lower() in ('yes', 'true', '1')
                            material = MaterialsModel.Material(self, mat_attrib.get('name', ''), base, properties,
                                                               alloy, mat.comments, props.get_comments())
                            new_entries.append(material)
                elif mat.tag in ('library', 'module'):
                    new_entries.append(MaterialsModel.External(self, mat.tag, mat.attrib.get('name', ''), mat.comments))
            self.endcomments = materials.get_comments()
        self._set_entries(new_entries, undoable)

    # XML element that represents whole section
    def make_xml_element(self):
        res = etree.Element(self.name)
        for e in self.entries:
            for c in e.comments:
                res.append(etree.Comment(c))
            res.append(e.make_xml_element())
        for c in self.endcomments:
            res.append(etree.Comment(c))
        return res

    def get_text(self):
        element = self.make_xml_element()
        if len(element) == 0: return ""
        lines = etree.tostring(element, pretty_print=True, encoding='unicode').splitlines()[1:-1]
        return "\n".join(line[2:] for line in lines)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if isinstance(self.entries[index.row()], MaterialsModel.External):
            if index.column() == 1:
                if role == Qt.ItemDataRole.FontRole:
                    font = QFont()
                    font.setItalic(True)
                    return font
            elif index.column() == 0 and role == Qt.ItemDataRole.DecorationRole:
                return QIcon.fromTheme(
                    {'library': 'material-library', 'module': 'material-module'}[self.entries[index.row()].what])
            elif index.column() == 2 and role == Qt.ItemDataRole.UserRole:
                return False
        if index.column() == 2:
            if role == Qt.ItemDataRole.UserRole:
                return True
            if role == Qt.ItemDataRole.ToolTipRole:
                return "Check this box if material is a generic alloy (i.e. an alloy material, which you can specify " \
                       "composition of).\nIts name must then consist of compound elements symbols with optional "\
                       "label and dopant, separated by '_' and ':' respectively"
        return super().data(index, role)

    def flags(self, index):
        flags = super().flags(index)
        if 1 <= index.column() < 3 and isinstance(self.entries[index.row()], MaterialsModel.External):
            flags &= ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsEnabled
        return flags

    def get(self, col, row):
        if col == 0: return self.entries[row].name
        if col == 1: return self.entries[row].base
        if col == 2: return self.entries[row].alloy
        if col == 3: return self.entries[row].comments
        raise IndexError(u'column number for MaterialsModel should be 0, 1, 2, or 3, but is {}'.format(col))

    def set(self, col, row, value):
        entry = self.entries[row]
        if col == 0: entry.name = value
        elif col == 1: entry.base = value
        elif col == 2: entry.alloy = value
        elif col == 3: entry.comments = value
        else: raise IndexError(u'column number for MaterialsModel should be 0, 1, 2, or 3, but is {}'.format(col))
        if isinstance(entry, MaterialsModel.External):
            entry.cache = None
            self.get_materials(row+1)
            index0 = entry.index(0, 0)
            last = len(entry.cache) - 1 if entry.cache is not None else 0
            index1 = entry.index(last, 0)
            entry.headerDataChanged.emit(Qt.Orientation.Vertical, 0, last)
            entry.dataChanged.emit(index0, index1)

    def create_default_entry(self):
        return MaterialsModel.Material(self, "name", "semiconductor")

    # QAbstractListModel implementation

    def columnCount(self, parent=QModelIndex()):
        return 3    # 4 if comment supported

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            if col == 0: return 'Name'
            if col == 1: return 'Base'
            if col == 2: return '(..)'
            if col == 3: return 'Comment'
        return None

    def create_info(self):
        res = super().create_info()

        names = {}
        for i, d in enumerate(self.entries):
            if isinstance(d, MaterialsModel.Material):
                if not d.name:
                    res.append(Info(u'Material name is required [row: {}]'.format(i+1), Info.ERROR, rows=[i], cols=[0]))
                else:
                    names.setdefault(d.name, []).append(i)
                if d.alloy:
                    name, label, groups, dope = parse_material_components(d.name, True)
                    elements = list(itertools.chain(*([e[0] for e in g] for g in groups)))
                    if len(''.join(elements)) != len(name):
                        res.append(Info(u"Alloy material's name does not consist of elements "
                                        u"and optional label [row: {}]"
                                        .format(i+1), Info.ERROR, rows=[i], cols=[0]))
                if not d.base:
                    res.append(Info(u'Material base is required [row: {}]'.format(i+1), Info.ERROR, rows=[i], cols=[1]))
                elif plask and d.base not in (e.name for e in self.entries[:i]) and '{' not in d.base:
                    try:
                        mat = str(d.base)
                        if ':'  in mat and '=' not in mat:
                            mat += '=0'
                        plask.material.db.get(mat)
                    except (ValueError, RuntimeError) as err:
                        if not(d.alloy and isinstance(err, ValueError) and
                               (str(err).startswith("Material composition required") or
                                str(err).startswith("Unknown material composition"))):
                            res.append(
                                Info(u"Material base '{1}' is not a proper material ({2}) [row: {0}]"
                                     .format(i+1, d.base, err), Info.ERROR, rows=[i], cols=[1]))
            else:
                if not d.name:
                    typ = {'library': 'Library', 'module': 'Module'}[d.what]
                    res.append(Info(u'{} name is required [row: {}]'.format(typ, i+1), Info.ERROR, rows=[i], cols=[0]))

        for name, rows in names.items():
            if len(rows) > 1:
                res.append(
                    Info(u'Duplicated material name "{}" [rows: {}]'.format(name, ', '.join(str(i+1) for i in rows)),
                         Info.ERROR, rows=rows, cols=[0]))
        return res

    def get_materials(self, limit=None):
        model_materials = []
        if limit is not None:
            entries = self.entries[:limit]
        else:
            entries = self.entries
        with self._HandleMaterialsModule(self.document) as module_handler:
            for material in entries:
                if isinstance(material, MaterialsModel.External):
                    if material.cache is None and plask is not None and material.name:
                        from .. import _DEBUG
                        if _DEBUG:
                            print("Reading materials from", material.what, material.name, file=sys.stderr)
                        if material.what == 'library':
                            try:
                                with plask.material.savedb(False):
                                    plask.material.load_library(material.name)
                                    material.cache = list(plask.material.db)
                            except RuntimeError:
                                if _DEBUG:
                                    import traceback
                                    traceback.print_exc()
                        elif material.what == 'module':
                            material.cache = module_handler.names = []
                            try:
                                with HandleMaterialsModule(self.document):
                                    if material.name in sys.modules:
                                        reload_module(sys.modules[material.name])
                                    else:
                                        import_module(material.name)
                            except:
                                if _DEBUG:
                                    import traceback
                                    traceback.print_exc()
                    if material.cache is not None:
                        model_materials = [m for m in model_materials if m not in material.cache] + material.cache
                else:
                    try: model_materials.remove(material.name)
                    except ValueError: pass
                    model_materials.append(material.name)
        return model_materials
