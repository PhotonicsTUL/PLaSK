from model.grids.grid import Mesh
from lxml.etree import ElementTree

class AxisConf(object):
    
    def __init__(self, start = None, stop = None, num = None, points = None):
        self.start = start
        self.stop = stop
        self.num = num
        self.points = points
        
    def addToXML(self, axisElement):
        if self.start: axisElement.attrib['start'] = self.start 
        if self.stop: axisElement.attrib['stop'] = self.stop
        if self.num: axisElement.attrib['num'] = self.num
        if self.points: axisElement.text = ", ".join(self.points)
        
class RectilinearMesh(Mesh):
    """Rectilinear Mesh model (1, 2, or 3D - see self.dim)"""
    
    def __init__(self, dim, name):
        Mesh.__init__(self, name)
        self.dim = dim
        self.axis = [AxisConf() for _ in range(0, dim)]
    
    @property
    def type(self):
        return "rectilinear%dd" % self.dim
    
    def addToXML(self, grids_section_element):
        res = Mesh.addToXML(self, grids_section_element)
        if self.dim == 1:
            self.axis[0].addToXML(ElementTree.SubElement(res, "axis"))
        else:
            for i, a in enumerate(self.axis):
                a.addToXML(ElementTree.SubElement(res, "axis%d" % i))
        return res;
    
    
    