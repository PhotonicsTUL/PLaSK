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

from . import GridTreeBased, GridWithoutConf
from .mesh_rectilinear import RectangularMesh, RectangularMesh1D
from .generator_rectilinear import RectilinearDivideGenerator

MESHES_TYPES = {
    'ordered': RectangularMesh1D.from_XML,
    'regular': RectangularMesh1D.from_XML,
    'rectangular2d': RectangularMesh.from_XML,
    'rectangular3d': RectangularMesh.from_XML,
}

GENERATORS_TYPES = {
    'ordered': {
            'divide': RectilinearDivideGenerator.from_XML,
            'simple': GridWithoutConf.from_XML,
    },
    'rectangular2d': {
            'divide': RectilinearDivideGenerator.from_XML,
            'simple': GridWithoutConf.from_XML,
    },
    'rectangular3d': {
            'divide': RectilinearDivideGenerator.from_XML,
            'simple': GridWithoutConf.from_XML,
    },
}   # use GridTreeBased.from_XML for grids without special support

DISPLAY_NAMES = {
    'rectilinear1d': "Rectilinear1D",
    'rectilinear2d': "Rectilinear2D",
    'rectilinear3d': "Rectilinear3D",
    'regular1d': "Regular1D",
    'regular2d': "Regular2D",
    'regular3d': "Regular3D",
}

def contruct_mesh(grids_model, element):
    t = MESHES_TYPES.get(element.attrib['type'])
    return t(grids_model, element) if t else GridTreeBased.from_XML(grids_model, element)

def contruct_generator(grids_model, element):
    t = GENERATORS_TYPES.get(element.attrib['type'])
    if t: t = t.get(element.attrib['method'])
    return t(grids_model, element) if t else GridTreeBased.from_XML(grids_model, element)


def construct_grid(grids_model, element):

    if element.tag == "mesh":
        k = element.attrib.keys()
        k.sort()
        if k != ['name', 'type']: raise ValueError('<mesh> tag must have two attributes (name and type), but has: %s' % ', '.join(k))
        return contruct_mesh(grids_model, element)

    if element.tag == "generator":
        k = element.attrib.keys()
        k.sort()
        if k != ['method', 'name', 'type']: raise ValueError('<generator> tag must have attributes "method", "name" and "type", but has: %s' % ', '.join(k))
        return contruct_generator(grids_model, element)

    raise ValueError('In <grids> section only <mesh> and <generator> tags are allowed, but got "%s".' % element.tag)


def display_name(item):
    ''':return: name of the mesh/generator to display in the GUI'''
    name = DISPLAY_NAMES.get(item)
    if name is None:
        name = item.title()
    return name

def xml_name(name):
    ''':return: XML tag name of the mesh/generator'''
    return name.lower() #TODO make it better if needed


def meshes_types():
    """:return: known types of meshes (list of strings)"""
    return (display_name(i) for i in MESHES_TYPES.keys())

def generators_types():
    """:return: known types of generator (list of strings)"""
    return (display_name(i) for i in GENERATORS_TYPES.keys())

def generator_methods(generator_name):
    """
        :param str generator_type: name of generator type
        :return: known methods for generator with given type (empty if the type of a generator is not known)
    """
    return (display_name(i) for i in GENERATORS_TYPES.get(generator_name, {}).keys())
