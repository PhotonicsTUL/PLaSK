from lxml.etree import Element, SubElement
from ...qt import QtCore
from ...model.table import TableModelEditMethods

from ...utils.xml import AttributeReader
from .grid import Grid

class RefinementConf(object):
    """Store refinement configuration of rectilinear generator"""

    attributes_names = ['object', 'path', 'at', 'by', 'every']
    all_attributes_names = ['axis'] + attributes_names

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
        if axis_element is None: return
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
        return RefinementConf()


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

    def __append_div_XML_element__(self, div_name, dst):
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
        self.__append_div_XML_element__('prediv', res)
        self.__append_div_XML_element__('postdiv', res)
        if len(self.refinements.entries) > 0:
            refinements_element = SubElement(res, 'refinements')
            for r in self.refinements.entries:
                refinements_element.append(r.get_XML_element)
        warnings_el = Element('warnings')
        for w in RectilinearDivideGenerator.warnings:
            v = getattr(self, 'warning_'+w, None)
            if v is not None and v != '': warnings_el.attrib[w] = v
        if warnings_el.attrib: res.append(warnings_el)
        return res

    def __div_from_XML__(self, div_name, src):
        div_element = src.find(div_name)
        if div_element is None:
            setattr(self, div_name, None)
        else:
            by = div_element.attrib.get('by')
            if by is not None:
                setattr(self, div_name, tuple(by for _ in range(0, self.dim)))
            else:
                setattr(self, div_name, tuple(div_element.attrib.get('by'+str(i)) for i in range(0, self.dim)))

    def set_XML_element(self, element):
        gradual_element = element.find('gradual')
        if gradual_element is not None:
            self.gradual = element.attrib.get('all', None)
        else:
            if element.find('no-gradual'):     #deprecated
                self.gradual = 'no'
            else:
                self.gradual = None
        self.__div_from_XML__('prediv', element)
        self.__div_from_XML__('postdiv', element)
        self.refinements.entries = []
        refinements_element = element.find('refinements')
        if refinements_element is not None:
            for ref_el in refinements_element:
                to_append = RefinementConf()
                to_append.set_from_XML(ref_el)
                self.refinements.entries.append(to_append)
        warnings_element = element.find('warnings')
        if warnings_element is None:
            for w in RectilinearDivideGenerator.warnings:
                setattr(self, 'warning_' + w, None)
        else:
            for w in RectilinearDivideGenerator.warnings:
                setattr(self, 'warning_' + w, warnings_element.attrib.get(w, None))

    def get_controller(self, document):
        from ...controller.grids.generator_rectilinear import RectilinearDivideGeneratorConroller
        return RectilinearDivideGeneratorConroller(document=document, model=self)