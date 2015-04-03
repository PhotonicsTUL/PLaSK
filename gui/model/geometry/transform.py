# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from .object import GNObject
from .constructor import construct_geometry_object
from ...utils.xml import xml_to_attr, attr_to_xml


class GNTransform(GNObject):

    def accept_as_child(self, node):
        from .geometry import GNGeometryBase
        if not self.accept_new_child() or isinstance(node, GNGeometryBase): return False
        from again_copy import GNCopy, GNAgain
        return (isinstance(node, GNObject) and node.dim == self.children_dim) or\
                isinstance(node, GNCopy) or isinstance(node, GNAgain)

    def _children_from_xml(self, ordered_reader, conf):
        construct_geometry_object(ordered_reader.get(), conf)

    def accept_new_child(self):
        return not self.children

    def real_to_model_index(self, path_iterator):
        path_iterator.next()
        return 0    #some transform, like mirror, can produce some fake, extra children


class GNTranslation(GNTransform):

    def __init__(self, parent=None, dim=None):
        super(GNTranslation, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.size = [None for _ in range(0, dim)]

    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNTranslation, self)._attributes_from_xml(attribute_reader, conf)
        axes_names = conf.axes_names(self.dim)
        alternative_names = ('depth', 'width', 'height')[(3-self.dim):]
        self.size = [None for _ in range(0, self.dim)]
        for i in range(0, self.dim):
            self.size[i] = attribute_reader.get('d' + axes_names[i])
            if self.size[i] is None:
                self.size[i] = attribute_reader.get(alternative_names[i])
                
    def _attributes_to_xml(self, element, conf):
        super(GNTranslation, self)._attributes_to_xml(element, conf)
        axes_names = conf.axes_names(self.dim)
        for i in range(0, self.dim):
            v = self.size[i]
            if v is not None: element.attrib['d' + axes_names[i]] = v

    def tag_name(self, full_name=True):
        return "translation{}d".format(self.dim) if full_name else "translation"

    def python_type(self):
        return 'geometry.Translation{}D'.format(self.dim)

    def major_properties(self):
        res = super(GNTranslation, self).major_properties()
        if any(self.size):
            res.append('delta', ', '.join(x if x else '?' for x in self.size))
        return res

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNTranslation(dim=2)
        result.set_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNTranslation(dim=3)
        result.set_xml_element(element, conf)
        return result


class GNClip(GNTransform):

    def __init__(self, parent=None, dim=None):
        super(GNClip, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.left = None
        self.right = None
        self.bottom = None
        self.top = None
        if dim == 3:
            self.back = None
            self.front = None

    def bound_names(self):
        d2 = ('left', 'right', 'bottom', 'top')
        return d2 if self.dim == 2 else d2 + ('back', 'front')

    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNClip, self)._attributes_from_xml(attribute_reader, conf)
        xml_to_attr(attribute_reader, self, *self.bound_names())

    def _attributes_to_xml(self, element, conf):
        super(GNClip, self)._attributes_to_xml(element, conf)
        attr_to_xml(self, element, *self.bound_names())

    def tag_name(self, full_name=True):
        return "clip{}d".format(self.dim) if full_name else "clip"

    def python_type(self):
        return 'geometry.Clip{}D'.format(self.dim)

    def major_properties(self):
        res = super(GNClip, self).major_properties()
        res.extend((n, getattr(self, n)) for n in self.bound_names())
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.transform import GNClipController
        return GNClipController(document, model, self)

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNClip(dim=2)
        result.set_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNClip(dim=3)
        result.set_xml_element(element, conf)
        return result


class GNAxisBaseTransform(GNTransform):
    '''Base class common for GNFlip and GNMirror. Includes axis attribute.'''

    def __init__(self, parent=None, dim=None):
        super(GNAxisBaseTransform, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.axis = None

    def set_axis(self, axis, conf = None):
        '''
            :param axis: new axis value (int or string)
            :param GNReadConf conf: configuration to get axes names from
        '''
        try:
            self.axis = (self.get_axes_conf_dim() if conf is None else conf.axes_names(self.dim)).index(axis)
        except Exception:
            self.axis = axis

    def axis_str(self, conf = None):
        '''
        Get axis as string.
        :param GNReadConf conf: configuration to get axes names from
        '''
        if self.axis is None or isinstance(self.axis, basestring): return self.axis
        if conf is not None: return conf.axis_name(self.dim, self.axis)
        return self.get_axes_conf_dim()[self.axis]

    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNAxisBaseTransform, self)._attributes_from_xml(attribute_reader, conf)
        self.set_axis(attribute_reader.get('axis'), conf)

    def _attributes_to_xml(self, element, conf):
        super(GNAxisBaseTransform, self)._attributes_to_xml(element, conf)
        a_str = self.axis_str(conf)
        if a_str is not None: element.attrib['axis'] = a_str

    def major_properties(self):
        res = super(GNAxisBaseTransform, self).major_properties()
        res.append(('axis', self.axis_str()))
        return res


class GNFlip(GNAxisBaseTransform):

    def tag_name(self, full_name=True):
        return "flip{}d".format(self.dim) if full_name else "flip"

    def python_type(self):
        return 'geometry.Flip{}D'.format(self.dim)

    def get_controller(self, document, model):
        from ...controller.geometry.transform import GNFlipController
        return GNFlipController(document, model, self)

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNFlip(dim=2)
        result.set_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNFlip(dim=3)
        result.set_xml_element(element, conf)
        return result


class GNMirror(GNAxisBaseTransform):

    def tag_name(self, full_name=True):
        return "mirror{}d".format(self.dim) if full_name else "mirror"

    def python_type(self):
        return 'geometry.Mirror{}D'.format(self.dim)

    def get_controller(self, document, model):
        from ...controller.geometry.transform import GNMirrorController
        return GNMirrorController(document, model, self)

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNMirror(dim=2)
        result.set_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNMirror(dim=3)
        result.set_xml_element(element, conf)
        return result


class GNIntersection(GNTransform):

    def __init__(self, parent=None, dim=None):
        super(GNIntersection, self).__init__(parent=parent, dim=dim, children_dim=dim)
        self.axis = None

    def tag_name(self, full_name=True):
        return "intersection{}d".format(self.dim) if full_name else "intersection"

    def python_type(self):
        return 'geometry.Intersection{}D'.format(self.dim)

    def accept_new_child(self):
        return len(self.children) < 2

    @staticmethod
    def from_xml_2d(element, conf):
        result = GNIntersection(dim=2)
        result.set_xml_element(element, conf)
        return result

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNIntersection(dim=3)
        result.set_xml_element(element, conf)
        return result


class GNExtrusion(GNTransform):

    def __init__(self, parent=None):
        super(GNExtrusion, self).__init__(parent=parent, dim=3, children_dim=2)
        self.length = None
        
    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNExtrusion, self)._attributes_from_xml(attribute_reader, conf)
        self.length = attribute_reader.get('length')

    def _attributes_to_xml(self, element, conf):
        super(GNExtrusion, self)._attributes_to_xml(element, conf)
        attr_to_xml(self, element, 'length')

    def tag_name(self, full_name=True):
        return "extrusion"

    def python_type(self):
        return 'geometry.Extrusion'

    def major_properties(self):
        res = super(GNExtrusion, self).major_properties()
        res.append(('length', self.length))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.transform import GNExtrusionController
        return GNExtrusionController(document, model, self)

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNExtrusion()
        result.set_xml_element(element, conf)
        return result


class GNRevolution(GNTransform):

    def __init__(self, parent=None):
        super(GNTransform, self).__init__(parent=parent, dim=3, children_dim=2)
        self.auto_clip = None

    def _attributes_from_xml(self, attribute_reader, conf):
        super(GNRevolution, self)._attributes_from_xml(attribute_reader, conf)
        self.auto_clip = attribute_reader.get('auto-clip')

    def _attributes_to_xml(self, element, conf):
        super(GNRevolution, self)._attributes_to_xml(element, conf)
        if self.auto_clip is not None: element.attrib['auto-clip'] = self.auto_clip

    def tag_name(self, full_name=True):
        return "revolution"

    def python_type(self):
        return 'geometry.Revolution'

    def major_properties(self):
        res = super(GNRevolution, self).major_properties()
        res.append(('auto-clip', self.auto_clip))
        return res

    def get_controller(self, document, model):
        from ...controller.geometry.transform import GNRevolutionController
        return GNRevolutionController(document, model, self)

    @staticmethod
    def from_xml_3d(element, conf):
        result = GNRevolution()
        result.set_xml_element(element, conf)
        return result