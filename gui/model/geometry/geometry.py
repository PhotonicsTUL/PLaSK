from .constructor import geometry_object_names, construct_geometry_object
from .object import GNObject
from .transform import GNExtrusion, GNRevolution
from ...utils.xml import xml_to_attr, attr_to_xml

class GNGeometryBase(GNObject):

    def __init__(self, parent=None, dim=None):
        super(GNGeometryBase, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.borders = [[None, None] for _ in range(0, dim)]

    def all_borders(self):
        """
            :return: border if all borders all are the same or None in other cases
        """
        b = self.borders[0]
        return b if all(x == b for x in self.borders) else None

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNGeometryBase, self).attributes_from_xml(attribute_reader, conf)
        all_names = self.get_alternative_direction_names()
        planar_names = all_names[:-1]

        b = attribute_reader.get('borders')
        if b is None:
            self.borders = [[None, None] for _ in range(0, self.dim)]
        else:
            self.borders = [[b, b] for _ in range(0, self.dim)]

        b = attribute_reader.get('planar')
        if b is not None:   #plana are all dirs except last (top/bottom)
            self.borders = [[b, b] for _ in range(0, self.dim-1)]

        for axis_nr, axis_name in enumerate(conf.axes_names(dim=self.dim)):
            for lo_hi_index, lo_or_hi in enumerate(('lo', 'hi')):
                a = attribute_reader.get(axis_name + '-' + lo_or_hi)
                alternative_name = all_names[axis_nr][lo_hi_index]
                b = attribute_reader.get(alternative_name)
                if a is not None:
                    if b is not None: raise ValueError("Border specified by both '{}' and '{}'.".format(axis_name + lo_or_hi, alternative_name))
                    self.borders[axis_nr][lo_hi_index] = a
                else:
                    if b is not None: self.borders[axis_nr][lo_hi_index] = b

    def attributes_to_xml(self, element, conf):
        super(GNGeometryBase, self).attributes_to_xml(element, conf)
        names = self.get_alternative_direction_names()
        for axis_nr in range(0, self.dim):
            for lo_hi_index in range(0, 2):
                val = self.borders[axis_nr][lo_hi_index]
                if val is not None: element.attrib[names[axis_nr][lo_hi_index]] = val

    def children_from_xml(self, ordered_reader, conf):
        construct_geometry_object(ordered_reader.get(), conf)

    def major_properties(self):
        res = super(GNGeometryBase, self).major_properties()
        b = self.all_borders()
        if b is not None:
            res.append(('all borders', b[1]))
        else:
            if not all(b is None for b in self.borders):
                res.append('borders:')
                names = self.get_alternative_direction_names()
                for axis_nr in range(0, self.dim):
                    for lo_hi_index in range(0, 2):
                        val = self.borders[axis_nr][lo_hi_index]
                        if val is not None: res.append((names[axis_nr][lo_hi_index], val))
                res.append(None)
        return res

    def accept_new_child(self):
        return not self.children


class GNCartesian(GNGeometryBase):

    def __init__(self, parent=None, dim=None):
        super(GNCartesian, self).__init__(parent=parent, dim=dim)
        if dim == 2:
            self.length = None

    def get_alternative_direction_names(self):
        planar = (('left', 'right'),)
        if self.dim == 3: planar = (('back', 'front'),) + planar
        return planar + (('bottom', 'top'),)

    def attributes_from_xml(self, attribute_reader, conf):
        super(GNCartesian, self).attributes_from_xml(attribute_reader, conf)
        if self.dim == 2:
            self.length = attribute_reader.get('length')

    def attributes_to_xml(self, element, conf):
        super(GNCartesian, self).attributes_to_xml(element, conf)
        if self.dim == 2:
            attr_to_xml(self, element, 'length')

    def children_from_xml(self, ordered_reader, conf):
        from .types import geometry_types_3d
        el = ordered_reader.get()
        if el is None: return
        if self.dim == 2 and el.tag in geometry_object_names(GNExtrusion.from_xml_3d, geometry_types_3d):
            #we don't require self.length is None, our model could store some "in building" states not well defined for PLaSK
            GNExtrusion.from_xml_3d(el, conf)
        else:
            construct_geometry_object(el, conf)

    def tag_name(self, full_name=True):
        return "cartesian{}d".format(self.dim)

    def python_type(self):
        return 'geometry.Cartesian{}D'.format(self.dim)

    def add_child_options(self):
        res = super(GNCartesian, self).add_child_options()
        if self.dim == 2:
            from .types import geometry_types_3d_core_extrusion
            res.insert(0, geometry_types_3d_core_extrusion)
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.geometry import GNGeometryController, GNCartesian2DGeometryController
        return GNCartesian2DGeometryController(document, model, self) if self.dim == 2 else GNGeometryController(document, model, self)

    @classmethod
    def from_xml_2d(cls, element, conf):
        result = GNCartesian(dim=2)
        result.set_xml_element(element, conf)
        return result

    @classmethod
    def from_xml_3d(cls, element, conf):
        result = GNCartesian(dim=3)
        result.set_xml_element(element, conf)
        return result


class GNCylindrical(GNGeometryBase):

    def __init__(self, parent=None):
        super(GNCylindrical, self).__init__(parent=parent, dim=2)

    def get_alternative_direction_names(self):
        return (('inner', 'outer'), ('bottom', 'top'))

    def children_from_xml(self, ordered_reader, conf):
        from .types import geometry_types_3d
        el = ordered_reader.get()
        if el is None: return
        if el.tag in geometry_object_names(GNRevolution.from_xml_3d, geometry_types_3d):
            #we don't require self.length is None, our model could store some "in building" states not well defined for PLaSK
            GNRevolution.from_xml_3d(el, conf)
        else:
            construct_geometry_object(el, conf)

    def tag_name(self, full_name=True):
        return "cylindrical{}d".format(self.dim) if full_name else "cylindrical"

    def python_type(self):
        return 'geometry.Cylindrical2D'

    def add_child_options(self):
        res = super(GNCylindrical, self).add_child_options()
        from .types import geometry_types_3d_core_revolution
        res.insert(0, geometry_types_3d_core_revolution)
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.geometry import GNGeometryController
        return GNGeometryController(document, model, self)

    @classmethod
    def from_xml_2d(cls, element, conf):
        result = GNCylindrical()
        result.set_xml_element(element, conf)
        return result