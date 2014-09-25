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

from lxml.etree import Element, SubElement
from ...qt import QtCore
from ...model.table import TableModelEditMethods

from ...utils.xml import AttributeReader, require_no_children, UnorderedTagReader
from .grid import Grid


class RefinementConf(object):
    """Store refinement configuration of rectilinear generator"""

    attributes_names = ['object', 'path', 'at', 'by', 'every']
    all_attributes_names = ['axis'] + attributes_names
    all_attributes_help = [
        'Name of axis where refinement should be added.',
        'Name of the geometry object to add additional division to. (required)',
        'Path name, specifying particular instance of the object given in the object attribute.',
        'If this is given, a single refinement line is placed at the position specified in it (in the local object coordinates).',
        'If this is given, multiple refinement lines are placed dividing the object into a specified number of equal parts.',
        'If this is given, multiple refinement lines are places at distance from each other specified by the value.'
     ]

    def __init__(self, axis=None, object=None, path=None, at=None, by=None, every=None):
        self.axis = axis
        self.object = object
        self.path = path
        self.at = at
        self.by = by
        self.every = every

    def get_XML_element(self):
        res = Element('axis{}'.format(self.axis))
        for attr in RefinementConf.attributes_names:
            a = getattr(self, attr, None)
            if a is not None: res.attrib[attr] = a
        return res

    def set_from_XML(self, axis_element):
        if axis_element is None:
            self.axis = 0
            for attr in RefinementConf.attributes_names: setattr(self, attr, None)
        else:
            require_no_children(axis_element)
            self.axis = int(axis_element.tag[-1])
            with AttributeReader(axis_element) as a:
                for attr in RefinementConf.attributes_names:
                    setattr(self, attr, a.get(attr, None))

    def get_attr_by_index(self, index):
        return getattr(self, RefinementConf.all_attributes_names[index])

    def set_attr_by_index(self, index, value):
        setattr(self, RefinementConf.all_attributes_names[index], int(value) if index == 0 else value)


class Refinements(QtCore.QAbstractTableModel, TableModelEditMethods):

    def __init__(self, generator, entries = None, parent=None, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        TableModelEditMethods.__init__(self)
        self.generator = generator
        self.entries = entries if entries is not None else []

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid(): return 0
        return len(self.entries)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(RefinementConf.all_attributes_names)

    def get(self, col, row):
        return self.entries[row].get_attr_by_index(col)

    def data(self, index, role = QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return self.get(index.column(), index.row())
        if role == QtCore.Qt.ToolTipRole:
            return RefinementConf.all_attributes_help[index.column()]

    def set(self, col, row, value):
        self.entries[row].set_attr_by_index(col, value)

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        self.set(index.column(), index.row(), value)
        self.dataChanged.emit(index, index)
        self.generator.fire_changed()
        return True

    def flags(self, index):
        flags = super(Refinements, self).flags(index) | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if not self.is_read_only(): flags |= QtCore.Qt.ItemIsEditable
        return flags

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            try:
                return RefinementConf.all_attributes_names[col]
            except IndexError:
                return None

    def is_read_only(self):
        return self.generator.is_read_only()

    def fire_changed(self):
        self.generator.fire_changed()

    def create_default_entry(self):
        return RefinementConf(axis=0)


class RectilinearDivideGenerator(Grid):
    """Model for all rectilinear generators (ordered, rectangular2d, rectangular3d)"""

    warnings = ('missing', 'multiple', 'outside')

    @staticmethod
    def from_XML(grids_model, element):
        e = RectilinearDivideGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.set_XML_element(element)
        return e

    def __init__(self, grids_model, name, type, gradual=False, prediv=None, postdiv=None, refinements=None,
                 warning_missing=None, warning_multiple=None, warning_outside=None):
        super(RectilinearDivideGenerator, self).__init__(grids_model, name, type, 'divide')
        self.gradual = gradual
        self.prediv = prediv
        self.postdiv = postdiv
        self.refinements = Refinements(self, refinements)
        self.warning_missing = warning_missing
        self.warning_multiple = warning_multiple
        self.warning_outside = warning_outside

    @property
    def dim(self):
        return 1 if self.type == 'ordered' else int(self.type[-2])

    def _append_div_XML_element(self, div_name, dst):
        div = getattr(self, div_name)
        if div is None: return
        div_element = Element(div_name)
        if div[0] is not None and div.count(div[0]) == self.dim:
            div_element.attrib['by'] = div[0]
            dst.append(div_element)
        else:
            for i in range(0, self.dim):
                if div[i] is not None: div_element.attrib['by' + str(i)] = div[i]
            if div_element.attrib:
                dst.append(div_element)

    def get_XML_element(self):
        res = super(RectilinearDivideGenerator, self).get_XML_element()
        if self.gradual is not None:
            SubElement(res, "gradual", attrib={'all': self.gradual})
        self._append_div_XML_element('prediv', res)
        self._append_div_XML_element('postdiv', res)
        if len(self.refinements.entries) > 0:
            refinements_element = SubElement(res, 'refinements')
            for r in self.refinements.entries:
                refinements_element.append(r.get_XML_element())
        warnings_el = Element('warnings')
        for w in RectilinearDivideGenerator.warnings:
            v = getattr(self, 'warning_'+w, None)
            if v is not None and v != '': warnings_el.attrib[w] = v
        if warnings_el.attrib: res.append(warnings_el)
        return res

    def _div_from_XML(self, div_name, src):
        div_element = src.find(div_name)
        if div_element is None:
            setattr(self, div_name, None)
        else:
            with AttributeReader(div_element) as a:
                by = a.get('by')
                if by is not None:
                    setattr(self, div_name, tuple(by for _ in range(0, self.dim)))
                else:
                    setattr(self, div_name, tuple(a.get('by'+str(i)) for i in range(0, self.dim)))

    def set_XML_element(self, element):
        super(RectilinearDivideGenerator, self).set_XML_element(element)
        with UnorderedTagReader(element) as r:
            gradual_element = r.find('gradual')
            if gradual_element is not None:
                with AttributeReader(gradual_element) as a: self.gradual = a.get('all', None)
            else:
                if r.find('no-gradual'):     #deprecated
                    self.gradual = 'no'
                else:
                    self.gradual = None
            self._div_from_XML('prediv', r)
            self._div_from_XML('postdiv', r)
            self.refinements.entries = []
            refinements_element = r.find('refinements')
            if refinements_element is not None:
                for ref_el in refinements_element:
                    to_append = RefinementConf()
                    to_append.set_from_XML(ref_el)
                    self.refinements.entries.append(to_append)
            warnings_element = r.find('warnings')
            if warnings_element is None:
                for w in RectilinearDivideGenerator.warnings:
                    setattr(self, 'warning_' + w, None)
            else:
                with AttributeReader(warnings_element) as a:
                    for w in RectilinearDivideGenerator.warnings:
                        setattr(self, 'warning_' + w, a.get(w, None))

    def _set_div(self, attr_name, div_tab):
        if div_tab is None or div_tab.count(None) == self.dim:
            setattr(self, attr_name, None)
        else:
            setattr(self, attr_name, div_tab)

    def get_prediv(self, index):
        return None if self.prediv is None else self.prediv[index]

    def set_prediv(self, prediv_tab):
        self._set_div('prediv', prediv_tab)

    def get_postdiv(self, index):
        return None if self.postdiv is None else self.postdiv[index]

    def set_postdiv(self, postdiv_tab):
        self._set_div('postdiv', postdiv_tab)

    def get_controller(self, document):
        from ...controller.grids.generator_rectilinear import RectilinearDivideGeneratorConroller
        return RectilinearDivideGeneratorConroller(document=document, model=self)