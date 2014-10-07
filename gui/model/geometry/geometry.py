from .object import GNObject
from .types import construct_geometry_object
from ...utils.xml import xml_to_attr

# TODO in case of cartesian 2d accept extrusion as a child
class GNCartesian(GNObject):

    def __init__(self, parent = None, dim = None):
        super(GNCartesian, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.left = None
        self.right = None
        self.bottom = None
        self.top = None
        if dim == 3:
            self.back = None
            self.front = None

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNCartesian, self).attributes_from_XML(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'left', 'right', 'bottom', 'top')
        if self.dim == 3:
            xml_to_attr(attribute_reader, self, 'back', 'front')


    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNCartesian(dim = 2)
        result.set_XML_element(element, conf)
        return result

    @classmethod
    def from_XML_3d(self, element, conf):
        result = GNCartesian(dim = 3)
        result.set_XML_element(element, conf)
        return result


class GNCylindrical(GNObject):

    def __init__(self, parent = None):
        super(GNCylindrical, self).__init__(parent=parent, dim=2, children_dim=2)
        self.bottom = None
        self.inner = None
        self.outer = None
        self.top = None

    def attributes_from_XML(self, attribute_reader, conf):
        super(GNCylindrical, self).attributes_from_XML(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'bottom', 'inner', 'outer', 'top')


    @classmethod
    def from_XML_2d(self, element, conf):
        result = GNCylindrical(dim = 2)
        result.set_XML_element(element, conf)
        return result