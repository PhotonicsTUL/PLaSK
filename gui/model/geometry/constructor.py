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

def geometry_object_names(constructor, *allowed_types):
    """:return: list of names"""
    if len(allowed_types) == 0:
        from .types import geometry_types_2d, geometry_types_3d, geometry_types_other
        return geometry_object_names(constructor, geometry_types_2d, geometry_types_3d, geometry_types_other)
    res = []
    for t in allowed_types:
        for n, c in t.items():
            if c == constructor: res.append(n)
    return res

def construct_geometry_object(element, conf, *allowed_types):
    if element is None: return None
    if len(allowed_types) == 0:
        from .types import geometry_types_2d, geometry_types_3d, geometry_types_other
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

def construct_by_name(type_name, *allowed_types):
    for m in allowed_types:
        c = m.get(type_name)
        if c is not None: return c(None, None)
    raise ValueError('Wrong name of geometry object type: {}'.format(type_name))