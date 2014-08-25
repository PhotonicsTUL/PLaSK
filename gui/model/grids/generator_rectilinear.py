from lxml.etree import Element

from ...utils.xml import AttributeReader
from .grid import Grid

class RefinementConf(object):
    """Store refinement configuration of rectilinear generator"""

    def __init__(self, axis_nr=None, object=None, path=None, at=None, by=None, every=None):
        self.axis_nr = axis_nr
        self.object = object
        self.path = path
        self.at = at
        self.by = by
        self.every = every

    def get_XML_element(self):
        res = Element('axis{}'.format(self.axis_nr))
        for attr in ['object', 'path', 'at', 'by', 'every']:
            a = getattr(self, attr, None)
            if a is not None: res.attrib[attr] = a
        return res

    def set_from_XML(self, axis_element):
        if axis_element is None: return
        self.axis_nr = int(axis_element.tag[-1])
        with AttributeReader(axis_element) as a:
            for attr in ['object', 'path', 'at', 'by', 'every']:
                setattr(self, attr, a.get(attr, None))