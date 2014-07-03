from lxml.etree import ElementTree, SubElement

from ...utils.xml import AttributeReader
from .grid import Grid


class AxisConf(object):
    """Store axis configuration of RectilinearMesh"""

    def __init__(self, start=None, stop=None, num=None, points=None, type=None):
        self.start = start
        self.stop = stop
        self.num = num
        self.points = points
        self.type = type

    def fillXMLElement(self, axisElement):
        for attr in ['start', 'stop', 'num', 'type']:
            a = getattr(self, attr, None)
            if a is not None: axisElement.attrib[attr] = a
        axisElement.text = self.points if self.points else ''

        #if self.start: axisElement.attrib['start'] = self.start
        #if self.stop: axisElement.attrib['stop'] = self.stop
        #if self.num: axisElement.attrib['num'] = self.num
        #if self.points: axisElement.text = ", ".join(self.points)
        #if self.points: axisElement.attrib['points'] = self.points

    def set_XML_element(self, axis_element):
        if axis_element is None or len(axis_element) == 0: return
        with AttributeReader(axis_element) as a:
            for attr in ['start', 'stop', 'num', 'type']:
                setattr(self, attr, a.get(attr, None))
        self.points = axis_element.text
        #self.points = [float(x) for x in axis_element.text.split(',')]


#RectangularMesh1D(Grid)
#if mesh_type in ('ordered', 'regular'):



class RectangularMesh(Grid):
    """Model of RectangularMesh (2D, or 3D - see self.dim)"""

    @staticmethod
    def from_XML(grids_model, element):
        e = RectangularMesh(grids_model, int(element.attrib['type'][-2]), element.attrib['name'])
        e.set_XML_element(element)
        return e

    def __init__(self, grids_model, dim, name):
        super(RectangularMesh, self).__init__(grids_model, name, 'rectangular{}d'.format(dim))
        self.dim = dim
        self.axis = [AxisConf() for _ in range(0, self.dim)]

    @staticmethod
    def axis_tag_name(nr):
        return "axis{}".format(nr)

    def get_XML_element(self):
        res = super(RectangularMesh, self).get_XML_element()
        for i in range(0, self.dim):
            self.axis[i].fillXMLElement(SubElement(res, RectangularMesh.axis_tag_name(i)))
        return res;

    def set_XML_element(self, element):
        for i in range(0, self.dim):
            self.axis[i].set_XML_element(element.find(RectangularMesh.axis_tag_name(i)))
