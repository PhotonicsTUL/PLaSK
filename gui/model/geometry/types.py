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

from .leaf import GNBlock, GNCircle, GNCylinder, GNTriangle
from .container import GNStack, GNAlignContainer
from .transform import GNClip, GNExtrusion, GNFlip, GNMirror, GNRevolution, GNTranslation
from .copy import GNCopy, GNAgain


geometry_types_2d = {

    # leafs:
    'block2d': GNBlock.from_xml_2d,
    'rectangle': GNBlock.from_xml_2d,
    'circle2d': GNCircle.from_xml_2d,
    'triangle': GNTriangle.from_xml_2d,

    # containers:
    'align2d': GNAlignContainer.from_xml_2d,
    'container2d': GNAlignContainer.from_xml_2d,
    'stack2d': GNStack.from_xml_2d,

    # transforms:
    'clip2d': GNClip.from_xml_2d,
    'flip2d': GNFlip.from_xml_2d,
    'mirror2d': GNMirror.from_xml_2d,
    'translation2d': GNTranslation.from_xml_2d
}


geometry_types_3d = {

    # leafs:
    'block3d': GNBlock.from_xml_3d,
    'cuboid': GNBlock.from_xml_3d,
    'circle3d': GNCircle.from_xml_3d,
    'sphere': GNCircle.from_xml_3d,
    'cylinder': GNCylinder.from_xml_3d,

    # containers:
    'align3d': GNAlignContainer.from_xml_3d,
    'container3d': GNAlignContainer.from_xml_3d,
    'stack3d': GNStack.from_xml_3d,

    # transforms:
    'clip3d': GNClip.from_xml_3d,
    'flip3d': GNFlip.from_xml_3d,
    'mirror3d': GNMirror.from_xml_3d,
    'translation3d': GNTranslation.from_xml_3d,
    'extrusion': GNExtrusion.from_xml_3d,
    'revolution': GNRevolution.from_xml_3d
}


geometry_types_other = {
    'again': GNAgain,
    'copy': GNCopy
}

def geometry_object_names(constructor, *allowed_types):
    """:return: list of names"""
    if len(allowed_types) == 0:
        return geometry_object_names(constructor, geometry_types_2d, geometry_types_3d, geometry_types_other)
    res = []
    for t in allowed_types:
        for n, c in t.items():
            if c == constructor: res.append(n)
    return res

def construct_geometry_object(element, conf, *allowed_types):
    if element is None: return None
    if len(allowed_types) == 0:
        d = conf.dim
        if d == 2: return construct_geometry_object(element, conf, geometry_types_2d, geometry_types_other)
        elif d == 3: return construct_geometry_object(element, conf, geometry_types_3d, geometry_types_other)
        else: return construct_geometry_object(element, conf, geometry_types_2d, geometry_types_3d, geometry_types_other)
    s = conf.suffix
    for m in allowed_types:
        c = m.get(element.tag)
        if c is None and s is not None: c = m.get(element.tag + s)
        if c is not None: return c(element, conf)
    raise ValueError('Unexpected tag: <{}>'.format(element.tag))
