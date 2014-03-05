# Base classes for entries in grids model
from xml.etree import ElementTree

class Grid(object):
    
    def __init__(self, name):
        object.__init__(self)
        
class Generator(Grid):
    
    def addToXML(self, grids_section_element):
        return ElementTree.SubElement(grids_section_element, "generator", { "name": self.name, "type": self.type, "method": self.method })

class Mesh(Grid):
    
    def addToXML(self, grids_section_element):
        return ElementTree.SubElement(grids_section_element, "mesh", { "name": self.name, "type": self.type })