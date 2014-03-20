from model.grids.grid import Grid
from lxml.etree import ElementTree

class AxisConf(object):
    """Store axis configuration of RectilinearMesh"""
    
    def __init__(self, start = None, stop = None, num = None, points = None):
        self.start = start
        self.stop = stop
        self.num = num
        self.points = points
        
    def fillXMLElement(self, axisElement):
        for attr in ['start', 'stop', 'num']:
            a = getattr(self, attr, None)
            if a != None: axisElement.attrib[attr] = a
        axisElement.text = self.points if self.points else ''
        
        #if self.start: axisElement.attrib['start'] = self.start 
        #if self.stop: axisElement.attrib['stop'] = self.stop
        #if self.num: axisElement.attrib['num'] = self.num
        #if self.points: axisElement.text = ", ".join(self.points)
        #if self.points: axisElement.attrib['points'] = self.points
        
    def set_XML_element(self, axis_element):
        if not axis_element: return
        for attr in ['start', 'stop', 'num']:
            setattr(self, attr, axis_element.attrib.get(attr, None))
        self.points = axis_element.text
        #self.points = [float(x) for x in axis_element.text.split(',')]
        
class RectilinearMesh(Grid):
    """Model of RectilinearMesh (1D, 2D, or 3D - see self.dim)"""
    
    @staticmethod
    def from_XML(grids_model, element):
        e = RectilinearMesh(grids_model, int(element.attrib['type'][-2]), element.attrib['name'])
        e.set_XML_element(element)
        return e
    
    def __init__(self, grids_model, dim, name):
        super(RectilinearMesh, self).__init__(grids_model, name, "rectilinear%dd" % dim)
        self.dim = dim
        self.axis = [AxisConf() for _ in range(0, dim)]
    
    @property
    def axes_index_name(self):
        if self.dim == 1:
            yield 0, 'axis'
        else:
            for i in range(0, self.dim): yield i, "axis%d" % i
    
    def get_XML_element(self):
        res = super(RectilinearMesh, self).get_XML_element()
        for i, n in self.axes_index_name:
            self.axis[i].fillXMLElement(ElementTree.SubElement(res, n))
        return res;
    
    def set_XML_element(self, element):
        for i, n in self.axis_index_name:
            self.axis[i].set_XML_element(element.find(n))
    