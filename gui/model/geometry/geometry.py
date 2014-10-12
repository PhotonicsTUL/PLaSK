from .object import GNObject
from .types import construct_geometry_object, geometry_object_names, geometry_types_3d
from .transform import GNExtrusion
from ...utils.xml import xml_to_attr


# TODO in case of cartesian 2d accept extrusion as a child
class GNCartesian(GNObject):

    def __init__(self, parent=None, dim=None):
        super(GNCartesian, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.left = None
        self.right = None
        self.bottom = None
        self.top = None
        if dim == 2:
            self.length = None
        else:
            self.back = None
            self.front = None

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNCartesian, self).attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'left', 'right', 'bottom', 'top')
        if self.dim == 2:
            self.length = attribute_reader.get('length')
        else:
            xml_to_attr(attribute_reader, self, 'back', 'front')

    def children_from_xml(self, ordered_reader, conf):
        el = ordered_reader.get()
        if el is None: return
        if self.dim == 2 and self.length is None and el.tag in geometry_object_names(GNExtrusion.from_xml_3d, geometry_types_3d):
            GNExtrusion.from_xml_3d(el, conf)
        else:
            construct_geometry_object(el, conf)

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNCartesian(dim=2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(self, element, conf):
        result = GNCartesian(dim = 3)
        result.set_xml_element(element, conf)
        return result


class GNCylindrical(GNObject):

    def __init__(self, parent=None):
        super(GNCylindrical, self).__init__(parent=parent, dim=2, children_dim=2)
        self.bottom = None
        self.inner = None
        self.outer = None
        self.top = None

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNCylindrical, self).attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, 'bottom', 'inner', 'outer', 'top')

    def children_from_xml(self, ordered_reader, conf):
        construct_geometry_object(ordered_reader.get(), conf)

    @classmethod
    def from_xml_2d(self, element, conf):
        result = GNCylindrical(dim = 2)
        result.set_xml_element(element, conf)
        return result