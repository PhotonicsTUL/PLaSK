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
        if self.start: axisElement.attrib['start'] = self.start 
        if self.stop: axisElement.attrib['stop'] = self.stop
        if self.num: axisElement.attrib['num'] = self.num
        if self.points: axisElement.text = ", ".join(self.points)
        
    def setXMLElement(self, axis_element):
        if not axis_element: return
        for attr in ['start', 'stop', 'num']:
            setattr(self, attr, axis_element.attrib.get(attr, None))
        self.points = [float(x) for x in axis_element.text.split(',')]
        
class RectilinearMesh(Grid):
    """Model of RectilinearMesh (1D, 2D, or 3D - see self.dim)"""
    
    def __init__(self, dim, name):
        super(RectilinearMesh, self).__init__(self, name, "rectilinear%dd" % dim)
        self.dim = dim
        self.axis = [AxisConf() for _ in range(0, dim)]
    
    @property
    def axes_index_name(self):
        if self.dim == 1:
            yield 0, 'axis'
        else:
            for i in range(0, self.dim): yield i, "axis%d" % i
    
    def getXMLElement(self):
        res = super(RectilinearMesh, self).getXMLElement()
        for i, n in self.axes_index_name:
            self.axis[i].fillXMLElement(ElementTree.SubElement(res, n))
        return res;
    
    def setXMLElement(self, element):
        for i, n in self.axis_index_name:
            self.axis[i].setXMLElement(element.find(n))
    