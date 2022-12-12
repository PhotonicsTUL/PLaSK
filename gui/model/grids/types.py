# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


from . import TreeFragmentGrid
from .mesh_rectangular import RectangularMesh, RectangularMesh1D
from .mesh_triangular import TriangularMesh2D, ExtrudedTriangularMesh3D
from .generator_rectangular import RectangularSimpleGenerator, RectangularRegularGenerator, \
    RectangularDivideGenerator, RectangularSmoothGenerator
from .generator_triangular import TriangularTriangleGenerator

MESHES_TYPES = {
    'ordered': RectangularMesh1D.from_xml,
    'regular': RectangularMesh1D.from_xml,
    'rectangular2d': RectangularMesh.from_xml,
    'rectangular3d': RectangularMesh.from_xml,
    'triangular2d': TriangularMesh2D.from_xml,
    'extruded_triangular3d': ExtrudedTriangularMesh3D.from_xml,
}

GENERATORS_TYPES = {
    'ordered': {
            'divide': RectangularDivideGenerator.from_xml,
            'regular': RectangularRegularGenerator.from_xml,
            'simple': RectangularSimpleGenerator.from_xml,
            'smooth': RectangularSmoothGenerator.from_xml,
    },
    'rectangular2d': {
            'divide': RectangularDivideGenerator.from_xml,
            'regular': RectangularRegularGenerator.from_xml,
            'simple': RectangularSimpleGenerator.from_xml,
            'smooth': RectangularSmoothGenerator.from_xml,
    },
    'rectangular3d': {
            'divide': RectangularDivideGenerator.from_xml,
            'regular': RectangularRegularGenerator.from_xml,
            'simple': RectangularSimpleGenerator.from_xml,
            'smooth': RectangularSmoothGenerator.from_xml,
    },
    'triangular2d': {
            'triangle': TriangularTriangleGenerator.from_xml
    }
}   # use TreeFragmentGrid.from_xml for grids without special support

DISPLAY_NAMES = {
    'rectilinear1d': "Rectilinear1D",
    'rectilinear2d': "Rectilinear2D",
    'rectilinear3d': "Rectilinear3D",
    'regular1d': "Regular1D",
    'regular2d': "Regular2D",
    'regular3d': "Regular3D",
    'triangular2d': "Triangular2D",
    'extruded_triangular3d': "ExtrudedTriangular3D",
}

def contruct_mesh(grids_model, element):
    t = MESHES_TYPES.get(element.attrib['type'])
    return t(grids_model, element) if t else TreeFragmentGrid.from_xml(grids_model, element)

def contruct_generator(grids_model, element):
    t = GENERATORS_TYPES.get(element.attrib['type'])
    if t: t = t.get(element.attrib['method'])
    return t(grids_model, element) if t else TreeFragmentGrid.from_xml(grids_model, element)


def construct_grid(grids_model, element):

    if element.tag == "mesh":
        k = element.attrib.keys()
        k.sort()
        if k != ['name', 'type']: raise ValueError('<mesh> tag must have two attributes (name and type), but has: %s' % ', '.join(k))
        result = contruct_mesh(grids_model, element)
        try:
            result.comments = element.comments
        except AttributeError:
            result.comments = []
        return result

    if element.tag == "generator":
        k = element.attrib.keys()
        k.sort()
        if k != ['method', 'name', 'type']: raise ValueError('<generator> tag must have attributes "method", "name" and "type", but has: %s' % ', '.join(k))
        result = contruct_generator(grids_model, element)
        try:
            result.comments = element.comments
        except AttributeError:
            result.comments = []
        return result

    raise ValueError('In <grids> section only <mesh> and <generator> tags are allowed, but got "%s".' % element.tag)


def display_name(item):
    """
    :return: name of the mesh/generator to display in the GUI
    """
    name = DISPLAY_NAMES.get(item)
    if name is None:
        name = item.title()
    return name


def xml_name(name):
    """
    :return: XML tag name of the mesh/generator
    """
    return name.lower() #TODO make it better if needed


def meshes_types():
    """
    :return: known types of meshes (list of strings)
    """
    return (display_name(i) for i in MESHES_TYPES.keys())


def generators_types():
    """
    :return: known types of generator (list of strings)
    """
    return (display_name(i) for i in GENERATORS_TYPES.keys())


def generator_methods(generator_name):
    """
    :param str generator_type: name of generator type
    :return: known methods for generator with given type (empty if the type of a generator is not known)
    """
    return (display_name(i) for i in GENERATORS_TYPES.get(generator_name, {}).keys())
