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
from ..table import TableModelEditMethods
from ...utils.xml import require_no_children, UnorderedTagReader, AttributeReader
from . import Grid
from .mesh_rectilinear import AXIS_NAMES


class RefinementConf(object):
    """Store refinement configuration of rectilinear generator"""

    attributes_names = ['object', 'path', 'at', 'by', 'every']
    all_attributes_names = ['axis'] + attributes_names
    all_attributes_help = [
        'Name of the axis where the refinement should be added.',
        'Name of the geometry object to add the additional division to. (required)',
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

    def get_xml_element(self):
        res = Element('axis{}'.format(self.axis))
        for attr in RefinementConf.attributes_names:
            a = getattr(self, attr, None)
            if a is not None: res.attrib[attr] = a
        return res

    def set_from_xml(self, axis_element):
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
        setattr(self, RefinementConf.all_attributes_names[index], value)


class Refinements(TableModelEditMethods, QtCore.QAbstractTableModel):

    def __init__(self, generator, entries=None, parent=None, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        TableModelEditMethods.__init__(self)
        self.generator = generator
        self.entries = entries if entries is not None else []
        self.one = int(self.generator.dim == 1)

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid(): return 0
        return len(self.entries)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(RefinementConf.all_attributes_names)-self.one

    def get(self, col, row):
        value = self.entries[row].get_attr_by_index(col+self.one)
        if not self.one and col == 0:
            value = AXIS_NAMES[self.generator.dim-1][value]
        return value

    get_raw = get

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
            return self.get(index.column(), index.row())
        if role == QtCore.Qt.ToolTipRole:
            col = index.column()
            attr = ('</b>... ', '<b>object</b>=""', '<b>path</b>=""',
                    '<b>at</b>=""', '<b>by</b>=""', '<b>every</b>=""')[col]
            return '&lt;{1}axis{0} {2}&gt;<br/>{3}'.format(
                self.get(0, index.row()), '<b>' if col == 0 else '', attr,
                RefinementConf.all_attributes_help[col])

    def set(self, col, row, value):
        if not self.one and col == 0:
            value = AXIS_NAMES[self.generator.dim-1].index(value)
        self.entries[row].set_attr_by_index(col+self.one, value)

    def flags(self, index):
        flags = super(Refinements, self).flags(index) | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if not self.is_read_only(): flags |= QtCore.Qt.ItemIsEditable
        return flags

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            try:
                return RefinementConf.all_attributes_names[col+self.one].title()
            except IndexError:
                return None

    def is_read_only(self):
        return self.generator.is_read_only()

    def fire_changed(self):
        self.generator.fire_changed()

    def create_default_entry(self):
        return RefinementConf(axis=0)

    @property
    def undo_stack(self):
        return self.generator.undo_stack


class RectilinearRefinedGenerator(Grid):

    warnings = ('missing', 'multiple', 'outside')

    def __init__(self, grids_model, name, type, method, aspect=None, refinements=None,
                 warn_missing=None, warn_multiple=None, warn_outside=None):

        super(RectilinearRefinedGenerator, self).__init__(grids_model, name, type, method)
        self.aspect = aspect
        self.refinements = Refinements(self, refinements)
        self.warn_missing = warn_missing
        self.warn_multiple = warn_multiple
        self.warn_outside = warn_outside

    @property
    def dim(self):
        return 1 if self.type == 'ordered' else int(self.type[-2])

    def get_xml_common(self, res, options=None):
        if options is None:
            options = {}
        if self.aspect:
            options['aspect'] = self.aspect
        if options:
            SubElement(res, "options", attrib=options)
        if len(self.refinements.entries) > 0:
            refinements_element = SubElement(res, 'refinements')
            for r in self.refinements.entries:
                refinements_element.append(r.get_xml_element())
        warnings_el = Element('warnings')
        for w in RectilinearDivideGenerator.warnings:
            v = getattr(self, 'warn_' + w, None)
            if v is not None and v != '': warnings_el.attrib[w] = v
        if warnings_el.attrib: res.append(warnings_el)

    def set_xml_common(self, res, *opts):
        options = res.find('options')
        if options is not None:
            with AttributeReader(options) as a:
                self.aspect = a.get('aspect', None)
                for opt in opts:
                    setattr(self, opt, a.get(opt, None))
        else:
            self.aspect = None
            for opt in opts:
                setattr(self, opt, None)
        self.refinements.entries = []
        refinements_element = res.find('refinements')
        if refinements_element is not None:
            for ref_el in refinements_element:
                to_append = RefinementConf()
                to_append.set_from_xml(ref_el)
                self.refinements.entries.append(to_append)
        warnings_element = res.find('warnings')
        if warnings_element is None:
            for w in RectilinearDivideGenerator.warnings:
                setattr(self, 'warn_' + w, None)
        else:
            with AttributeReader(warnings_element) as a:
                for w in RectilinearDivideGenerator.warnings:
                    setattr(self, 'warn_' + w, a.get(w, None))


class RectilinearDivideGenerator(RectilinearRefinedGenerator):
    """Model for all rectilinear generators (ordered, rectangular2d, rectangular3d)"""

    @staticmethod
    def from_xml(grids_model, element):
        e = RectilinearDivideGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.set_xml_element(element)
        return e

    def __init__(self, grids_model, name, type, gradual=None, aspect=None, prediv=None, postdiv=None, refinements=None,
                 warn_missing=None, warn_multiple=None, warn_outside=None):

        super(RectilinearDivideGenerator, self).__init__(grids_model, name, type, 'divide', aspect,
                                                         refinements, warn_missing, warn_multiple, warn_outside)
        self.gradual = gradual
        self.prediv = prediv
        self.postdiv = postdiv

    def _append_div_xml_element(self, div_name, dst):
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

    def get_xml_element(self):
        res = super(RectilinearDivideGenerator, self).get_xml_element()
        options = {}
        if self.gradual is not None:
            options['gradual'] = self.gradual
        self._append_div_xml_element('prediv', res)
        self._append_div_xml_element('postdiv', res)
        self.get_xml_common(res, options)
        return res

    def _div_from_xml(self, div_name, src):
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

    def set_xml_element(self, element):
        super(RectilinearDivideGenerator, self).set_xml_element(element)
        with UnorderedTagReader(element) as res:
            self._div_from_xml('prediv', res)
            self._div_from_xml('postdiv', res)
            self.set_xml_common(res, 'gradual')

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
        from ...controller.grids.generator_rectilinear import RectilinearDivideGeneratorController
        return RectilinearDivideGeneratorController(document=document, model=self)


class RectilinearSmoothGenerator(RectilinearRefinedGenerator):
    """Model for all rectilinear generators (ordered, rectangular2d, rectangular3d)"""

    @staticmethod
    def from_xml(grids_model, element):
        e = RectilinearSmoothGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.set_xml_element(element)
        return e

    def __init__(self, grids_model, name, type, aspect=None, small=None, large=None, factor=None, refinements=None,
                 warn_missing=None, warn_multiple=None, warn_outside=None):

        super(RectilinearSmoothGenerator, self).__init__(grids_model, name, type, 'smooth', aspect,
                                                         refinements, warn_missing, warn_multiple, warn_outside)
        self.small = small
        self.large = large
        self.factor = factor

    @property
    def dim(self):
        return 1 if self.type == 'ordered' else int(self.type[-2])

    def _set_steps_attribute(self, attr, element):
        value = getattr(self, attr)
        if value is None: return
        if value[0] is not None and value.count(value[0]) == self.dim:
            element.attrib[attr] = value[0]
        else:
            for i in range(0, self.dim):
                if value[i] is not None:
                    element.attrib[attr + str(i)] = value[i]

    def get_xml_element(self):
        res = super(RectilinearSmoothGenerator, self).get_xml_element()
        if self.small is not None or self.factor is not None:
            steps = SubElement(res, 'steps')
            self._set_steps_attribute('small', steps)
            self._set_steps_attribute('large', steps)
            self._set_steps_attribute('factor', steps)
        self.get_xml_common(res)
        return res

    def _steps_from_xml(self, name, reader):
        val = reader.get(name)
        if val is not None:
            setattr(self, name, self.dim * (val,))
        else:
            value = tuple(reader.get(name + str(i)) for i in range(0, self.dim))
            if value == self.dim * (None,): value = None
            setattr(self, name, value)

    def set_xml_element(self, element):
        super(RectilinearSmoothGenerator, self).set_xml_element(element)
        with UnorderedTagReader(element) as res:
            self.set_xml_common(res, 'gradual')
            steps = res.find('steps')
            if steps is None:
                self.small = self.factor = None
            else:
                with AttributeReader(steps) as a:
                    self._steps_from_xml('small', a)
                    self._steps_from_xml('large', a)
                    self._steps_from_xml('factor', a)

    def _set_step(self, name, tab):
        if tab is None or tab.count(None) == self.dim:
            setattr(self, name, None)
        else:
            setattr(self, name, tab)

    def get_small(self, index):
        return None if self.small is None else self.small[index]

    def set_small(self, small):
        self._set_step('small', small)

    def get_large(self, index):
        return None if self.large is None else self.large[index]

    def set_large(self, large):
        self._set_step('large', large)

    def get_factor(self, index):
        return None if self.factor is None else self.factor[index]

    def set_factor(self, factor):
        self._set_step('factor', factor)

    def get_controller(self, document):
        from ...controller.grids.generator_rectilinear import RectilinearSmoothGeneratorController
        return RectilinearSmoothGeneratorController(document=document, model=self)
