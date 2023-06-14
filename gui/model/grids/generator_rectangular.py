# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


from lxml import etree

from ...qt.QtCore import *
from ..table import TableModelEditMethods
from ...utils.str import empty_to_none
from ...utils.validators import can_be_float, can_be_bool, can_be_int
from ...utils.xml import require_no_children, UnorderedTagReader, AttributeReader, OrderedTagReader
from . import Grid
from .mesh_rectangular import AXIS_NAMES


class RefinementConf:
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
        self.comments = []

    def make_xml_element(self):
        res = etree.Element('axis{}'.format(self.axis))
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


class Refinements(TableModelEditMethods, QAbstractTableModel):

    def __init__(self, generator, entries=None, parent=None, *args):
        QAbstractTableModel.__init__(self, parent, *args)
        TableModelEditMethods.__init__(self)
        self.generator = generator
        self.entries = entries if entries is not None else []
        self.one = int(self.generator.dim == 1)

    @property
    def name(self):
        try:
            generator_name = self.generator.name + ' '
        except AttributeError:
            generator_name = ''
        return "{}refinements".format(generator_name)

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid(): return 0
        return len(self.entries)

    def columnCount(self, parent=QModelIndex()):
        return len(RefinementConf.all_attributes_names)-self.one

    def get(self, col, row):
        value = self.entries[row].get_attr_by_index(col+self.one)
        if not self.one and col == 0:
            value = AXIS_NAMES[self.generator.dim-1][value]
        return value

    get_raw = get

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid(): return None
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            return self.get(index.column(), index.row())
        if role == Qt.ItemDataRole.ToolTipRole:
            col = index.column()
            attr = ('</b>... ', '<b>object</b>=""', '<b>path</b>=""',
                    '<b>at</b>=""', '<b>by</b>=""', '<b>every</b>=""')[col]
            return '&lt;{1}axis{0} {2}&gt;<br/>{3}'.format(
                self.get(0, index.row()), '<b>' if col == 0 else '', attr,
                RefinementConf.all_attributes_help[col])

    def set(self, col, row, value):
        if not self.one and col == 0:
            value = AXIS_NAMES[self.generator.dim-1].index(value)
            self.entries[row].set_attr_by_index(0, value)
        else:
            self.entries[row].set_attr_by_index(col+self.one, empty_to_none(value))

    def flags(self, index):
        flags = super().flags(index) | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if not self.is_read_only(): flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
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

    def create_info(self, res, rows, parent_property):
        for i, entry in enumerate(self.entries):
            if not entry.object:
                self.generator._required(res, rows, parent_property, 'name of the geometry object in refinement',
                                         reinf_row=i, reinf_col=1)
            if not can_be_float(entry.at):
                self.generator._required(res, rows, parent_property, 'position of a line in refinement', type='float',
                                         reinf_row=i, reinf_col=3)
            if not can_be_int(entry.by, int_validator=lambda n: n > 0):
                self.generator._required(res, rows, parent_property, 'number of division parts in refinement',
                                         type='positive integer', reinf_row=i, reinf_col=4)
            if not can_be_float(entry.every, float_validator=lambda f: f >= 0):
                self.generator._required(res, rows, parent_property, 'distance between lines in refinement',
                                         type='non-negative float', reinf_row=i, reinf_col=5)


class RectangularSimpleGenerator(Grid):

    @staticmethod
    def from_xml(grids_model, element):
        e = RectangularSimpleGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.load_xml_element(element)
        return e

    def __init__(self, grids_model, name, type, method='simple'):
        super().__init__(grids_model, name, type, method)
        self.split = None
        self.boundaries_comments = []

    @property
    def dim(self):
        return 1 if self.type == 'ordered' else int(self.type[-2])

    def make_xml_element(self):
        res = super().make_xml_element()
        for c in self.boundaries_comments:
            res.append(etree.Comment(c))
        if self.split is not None:
            etree.SubElement(res, "boundaries", attrib={'split': self.split})
        self.save_endcomments(res)
        return res

    def _read_xml(self, reader):
        boundaries = reader.find('boundaries')
        if boundaries is not None:
            self.split = boundaries.attrib.get('split')
            self.boundaries_comments = boundaries.comments
        else:
            self.boundaries_comments = []
        self.endcomments = reader.get_comments()

    def load_xml_element(self, element):
        if isinstance(element, UnorderedTagReader):
            super().load_xml_element(element.parent_element)
            self._read_xml(element)
        else:
            super().load_xml_element(element)
            with UnorderedTagReader(element) as reader:
                self._read_xml(reader)

    def get_controller(self, document):
        from ...controller.grids.generator_rectangular import RectangularSimpleGeneratorController
        return RectangularSimpleGeneratorController(document=document, model=self)

    def create_info(self, res, rows):
        super().create_info(res, rows)
        if not can_be_bool(self.split):
            self._required(res, rows, 'split', type='bool')


class RectangularRegularGenerator(RectangularSimpleGenerator):

    @staticmethod
    def from_xml(grids_model, element):
        e = RectangularRegularGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.load_xml_element(element)
        return e

    def __init__(self, grids_model, name, type):
        super().__init__(grids_model, name, type, 'regular')
        self.spacing0 = None
        self.spacing1 = None
        self.spacing2 = None
        self.spacing_comments = []

    def make_xml_element(self):
        res = super().make_xml_element()
        dim = self.dim
        if dim == 1:
            if self.spacing0 is not None:
                attrs = {'every': self.spacing0}
            else:
                attrs = {}
        else:
            attrs = {}
            for i in range(dim):
                attr = getattr(self,'spacing{}'.format(i))
                if attr is not None:
                    attrs['every{}'.format(i)] = attr
        for c in self.spacing_comments:
            res.append(etree.Comment(c))
        if attrs:
            etree.SubElement(res, "spacing", attrib=attrs)
        return res

    def load_xml_element(self, element):
        with UnorderedTagReader(element) as reader:
            super().load_xml_element(reader)
            spacing = reader.find('spacing')
            if spacing is not None:
                dim = self.dim
                if dim == 1:
                    self.spacing0 = spacing.attrib.get('every')
                else:
                    for i in range(dim):
                        setattr(self,'spacing{}'.format(i), spacing.attrib.get('every{}'.format(i)))
                self.spacing_comments = spacing.comments
            else:
                self.spacing_comments = []

    def get_controller(self, document):
        from ...controller.grids.generator_rectangular import RectangularRegularGeneratorController
        return RectangularRegularGeneratorController(document=document, model=self)

    def create_info(self, res, rows):
        super().create_info(res, rows)
        dim = self.dim
        for i in range(dim):
            if not can_be_float(getattr(self, 'spacing{}'.format(i))):
                self._required(res, rows, 'spacing{}'.format(i), type='float')

class RectangularRefinedGenerator(Grid):

    def __init__(self, grids_model, name, type, method, aspect=None, refinements=None):
        super().__init__(grids_model, name, type, method)
        self.aspect = aspect
        self.refinements = Refinements(self, refinements)
        self.options_comments = []
        self.refinements_comments = []
        self.refinements_endcomments = []

    @property
    def dim(self):
        return 1 if self.type == 'ordered' else int(self.type[-2])

    def save_xml_common(self, res, options=None):
        if options is None:
            options = {}
        if self.aspect:
            options['aspect'] = self.aspect
        for c in self.options_comments:
            res.append(etree.Comment(c))
        if options:
            etree.SubElement(res, "options", attrib=options)
        for c in self.refinements_comments:
            res.append(etree.Comment(c))
        if len(self.refinements.entries) > 0:
            refinements_element = etree.SubElement(res, 'refinements')
            for r in self.refinements.entries:
                for c in r.comments:
                    refinements_element.append(etree.Comment(c))
                refinements_element.append(r.make_xml_element())
            for c in self.refinements_endcomments:
                refinements_element.append(etree.Comment(c))

    def load_xml_common(self, reader: UnorderedTagReader, *opts):
        reader.mark_read('warnings')
        options_element = reader.find('options')
        if options_element is not None:
            with AttributeReader(options_element) as a:
                self.aspect = a.get('aspect')
                if 'gradual' in a:
                    g = a['gradual']
                    self.gradual = [g for _ in range(0, self.dim)]
                for i in range(self.dim):
                    g = a.get('gradual{}'.format(i))
                    if g is not None:
                        self.gradual[i] = g
                for opt in opts:
                    setattr(self, opt, a.get(opt))
            self.options_comments = options_element.comments
        else:
            self.aspect = None
            for opt in opts:
                setattr(self, opt, None)
            self.options_comments = []
        self.refinements.entries = []
        refinements_element = reader.find('refinements')
        if refinements_element is not None:
            with OrderedTagReader(refinements_element) as refinements_reader:
                for ref_el in refinements_reader:
                    to_append = RefinementConf()
                    to_append.set_from_xml(ref_el)
                    to_append.comments = ref_el.comments
                    self.refinements.entries.append(to_append)
                self.refinements_endcomments = refinements_reader.get_comments()
            self.refinements_comments = refinements_element.comments
        else:
            self.refinements_comments = []

    def create_info(self, res, rows):
        super().create_info(res, rows)
        self.refinements.create_info(res, rows, 'refinements')
        if not can_be_float(self.aspect): self._required(res, rows, 'aspect', type='float')


class RectangularDivideGenerator(RectangularRefinedGenerator):
    """Model for all rectilinear generators (ordered, rectangular2d, rectangular3d)"""

    @staticmethod
    def from_xml(grids_model, element):
        e = RectangularDivideGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.load_xml_element(element)
        return e

    def __init__(self, grids_model, name, type, gradual=None, aspect=None, prediv=None, postdiv=None, refinements=None):

        super().__init__(grids_model, name, type, 'divide', aspect, refinements)
        self.gradual = [None for _ in range(0, self.dim)] if gradual is None else gradual
        self.prediv = [None for _ in range(0, self.dim)] if prediv is None else prediv
        self.postdiv = [None for _ in range(0, self.dim)] if postdiv is None else postdiv
        self.prediv_comments = []
        self.postdiv_comments = []

    def _append_div_xml_element(self, div_name, dst):
        div = getattr(self, div_name)
        for c in getattr(self, div_name + '_comments'):
            dst.append(etree.Comment(c))
        #if div is None: return
        div_element = etree.Element(div_name)
        if div[0] is not None and div.count(div[0]) == self.dim:
            div_element.attrib['by'] = div[0]
            dst.append(div_element)
        else:
            for i in range(0, self.dim):
                if div[i] is not None: div_element.attrib['by' + str(i)] = div[i]
            if div_element.attrib:
                dst.append(div_element)

    def make_xml_element(self):
        res = super().make_xml_element()
        options = {}
        if self.dim == 1 or all(g == self.gradual[0] for g in self.gradual):
            if self.gradual[0] is not None:
                options['gradual'] = self.gradual[0]
        else:
            for i in range(self.dim):
                if self.gradual[i] is not None:
                    options['gradual{}'.format(i)] = self.gradual[i]
        self._append_div_xml_element('prediv', res)
        self._append_div_xml_element('postdiv', res)
        self.save_xml_common(res, options)
        self.save_endcomments(res)
        return res

    def _div_from_xml(self, div_name, src):
        div_element = src.find(div_name)
        if div_element is None:
            setattr(self, div_name, [None for _ in range(0, self.dim)])
            setattr(self, div_name + '_comments', [])
        else:
            with AttributeReader(div_element) as a:
                by = a.get('by')
                if by is not None:
                    setattr(self, div_name, [by for _ in range(0, self.dim)])
                else:
                    setattr(self, div_name, [a.get('by'+str(i)) for i in range(0, self.dim)])
            setattr(self, div_name + '_comments', div_element.comments)

    def load_xml_element(self, element):
        super().load_xml_element(element)
        with UnorderedTagReader(element) as reader:
            self._div_from_xml('prediv', reader)
            self._div_from_xml('postdiv', reader)
            self.load_xml_common(reader)
            self.endcomments = reader.get_comments()

    def get_controller(self, document):
        from ...controller.grids.generator_rectangular import RectangularDivideGeneratorController
        return RectangularDivideGeneratorController(document=document, model=self)

    def create_info(self, res, rows):
        super().create_info(res, rows)
        for div_type in ('prediv', 'postdiv'):
            for i, p in enumerate(getattr(self, div_type)):
                if not can_be_int(p, int_validator=lambda n: n>0):
                    self._required(res, rows, (div_type, i), 'a component of {}'.format(div_type), type='positive integer')
        for i in range(self.dim):
            if not can_be_bool(self.gradual[i]):
                self._required(res, rows, 'gradual{}'.format(i), type='boolean')


class RectangularSmoothGenerator(RectangularRefinedGenerator):
    """Model for all rectilinear generators (ordered, rectangular2d, rectangular3d)"""

    @staticmethod
    def from_xml(grids_model, element):
        e = RectangularSmoothGenerator(grids_model, element.attrib['name'], element.attrib['type'])
        e.load_xml_element(element)
        return e

    def __init__(self, grids_model, name, type, aspect=None, small=None, large=None, factor=None, refinements=None):

        super().__init__(grids_model, name, type, 'smooth', aspect, refinements)
        self.small = [None for _ in range(0, self.dim)] if small is None else small
        self.large = [None for _ in range(0, self.dim)] if large is None else large
        self.factor = [None for _ in range(0, self.dim)] if factor is None else factor
        self.steps_comments = []

    @property
    def dim(self):
        return 1 if self.type == 'ordered' else int(self.type[-2])

    def _set_steps_attribute(self, attr, element):
        value = getattr(self, attr)
        #if value is None: return
        if value[0] is not None and value.count(value[0]) == self.dim:
            element.attrib[attr] = value[0]
        else:
            for i in range(0, self.dim):
                if value[i] is not None:
                    element.attrib[attr + str(i)] = value[i]

    def make_xml_element(self):
        res = super().make_xml_element()
        for c in self.steps_comments:
            res.append(etree.Comment(c))
        if self.small is not None or self.factor is not None:
            steps = etree.SubElement(res, 'steps')
            self._set_steps_attribute('small', steps)
            self._set_steps_attribute('large', steps)
            self._set_steps_attribute('factor', steps)
        self.save_xml_common(res)
        self.save_endcomments(res)
        return res

    def _steps_from_xml(self, name, reader):
        val = reader.get(name)
        if val is not None:
            setattr(self, name, self.dim * [val,])
        else:
            setattr(self, name, [reader.get(name + str(i)) for i in range(0, self.dim)])

    def load_xml_element(self, element):
        super().load_xml_element(element)
        with UnorderedTagReader(element) as reader:
            self.load_xml_common(reader)
            steps = reader.find('steps')
            if steps is None:
                self.small = self.dim * [None]
                self.factor = self.dim * [None]
                self.steps_comments = []
            else:
                with AttributeReader(steps) as a:
                    self._steps_from_xml('small', a)
                    self._steps_from_xml('large', a)
                    self._steps_from_xml('factor', a)
                self.steps_comments = steps.comments
            self.endcomments = reader.get_comments()

    def create_info(self, res, rows):
        super().create_info(res, rows)
        for i, p in enumerate(self.small):
            if not can_be_float(p): self._required(res, rows, ('small', i), 'a component of smallest element', type='float')
        for i, p in enumerate(self.large):
            if not can_be_float(p): self._required(res, rows, ('large', i), 'a component of largest element', type='float')
        for i, p in enumerate(self.factor):
            if not can_be_float(p): self._required(res, rows, ('factor', i), 'a component of increase factor', type='float')

    def get_controller(self, document):
        from ...controller.grids.generator_rectangular import RectangularSmoothGeneratorController
        return RectangularSmoothGeneratorController(document=document, model=self)
