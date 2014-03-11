# Base classes for entries in grids model
from lxml.etree import ElementTree

class Grid(object):
    """Base class for models of grids (meshes or generators)"""
    
    def __init__(self, name):
        object.__init__(self)
        
class Generator(Grid):
    """Base class for models of generators"""
    
    def addToXML(self, grids_section_element):
        return ElementTree.SubElement(grids_section_element, "generator", { "name": self.name, "type": self.type, "method": self.method })

class Mesh(Grid):
    """Base class for models of meshes"""
    
    def addToXML(self, grids_section_element):
        return ElementTree.SubElement(grids_section_element, "mesh", { "name": self.name, "type": self.type })