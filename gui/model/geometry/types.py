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


GNAMES = {
    'circle': "Circle",

    'align': "Align",
    'stack': "Stack",
    'shelf': "Shelf",

    'clip': "Clip",
    'flip': "Flip",
    'mirror': "Mirror",
    'translation': "Translation",
    'intersection': "Intersection",

    'block': "Block",


    'rectangle': "Rectangle",
    'circle2d': "Circle2D",
    'triangle': "Triangle",

    'align2d': "Align2D",
    'stack2d': "Stack2D",
    'shelf2d': "Shelf2D",

    'clip2d': "Clip2D",
    'flip2d': "Flip2D",
    'mirror2d': "Mirror2D",
    'translation2d': "Translation2D",
    'intersection2d': "Intersection2D",

    'block2d': "Block2D",

    'cuboid': "Cuboid",
    'sphere': "Sphere",
    'cylinder': "Cylinder",

    'align3d': "Align3D",
    'stack3d': "Stack3D",

    'extrusion': "Extrusion",
    'revolution': "Revolution",

    'clip3d': "Clip3D",
    'flip3d': "Flip3D",
    'mirror3d': "mirror3D",
    'translation3d': "translation3D",
    'intersection3d': "intersection3D",

    'block3d': "Block3D",
    'circle3d': "Circle3D",

    'again': "Again",
    'copy': "Copy",

    'cartesian2d': "Cartesian2D",
    'cartesian3d': "Cartesian3D",
    'cylindrical': "Cylindrical",
    'cylindrical2d': "Cylindrical2D",
}


def gname(key):
    return GNAMES.get(key, key)


from .leaf import GNBlock, GNCircle, GNCylinder, GNTriangle
from .container import GNStack, GNAlignContainer, GNShelf
from .transform import GNClip, GNExtrusion, GNFlip, GNMirror, GNRevolution, GNTranslation, GNIntersection
from .copy import GNCopy, GNAgain
from .geometry import GNCartesian, GNCylindrical

geometry_types_2d_core_leafs = {  # only unique names of types, displayed in add menu
    'rectangle': GNBlock.from_xml_2d,
    'circle2d': GNCircle.from_xml_2d,
    'triangle': GNTriangle.from_xml_2d
}
geometry_types_2d_core_containers = {  # only unique names of types, displayed in add menu
    'align2d': GNAlignContainer.from_xml_2d,
    'stack2d': GNStack.from_xml_2d,
    'shelf2d': GNShelf.from_xml_2d
}
geometry_types_2d_core_transforms = {  # only unique names of types, displayed in add menu
    'clip2d': GNClip.from_xml_2d,
    'flip2d': GNFlip.from_xml_2d,
    'mirror2d': GNMirror.from_xml_2d,
    'translation2d': GNTranslation.from_xml_2d,
    'intersection2d': GNIntersection.from_xml_2d
}

geometry_types_2d_core = {}  # only unique names of types, displayed in add menu
geometry_types_2d_core.update(geometry_types_2d_core_leafs)
geometry_types_2d_core.update(geometry_types_2d_core_containers)
geometry_types_2d_core.update(geometry_types_2d_core_transforms)

geometry_types_2d = {   # all names: geometry_types_2d_core + aliases

    # leafs:
    'block2d': GNBlock.from_xml_2d,

    # containers:
    'container2d': GNAlignContainer.from_xml_2d,

    # transforms:
}
geometry_types_2d.update(geometry_types_2d_core)

geometry_types_3d_core_leafs = {
    'cuboid': GNBlock.from_xml_3d,
    'sphere': GNCircle.from_xml_3d,
    'cylinder': GNCylinder.from_xml_3d,
}
geometry_types_3d_core_containers = {
    'align3d': GNAlignContainer.from_xml_3d,
    'stack3d': GNStack.from_xml_3d,
}
geometry_types_3d_core_extrusion = {
    'extrusion': GNExtrusion.from_xml_3d
}
geometry_types_3d_core_revolution = {
    'revolution': GNRevolution.from_xml_3d
}
geometry_types_3d_core_transforms = {
    'clip3d': GNClip.from_xml_3d,
    'flip3d': GNFlip.from_xml_3d,
    'mirror3d': GNMirror.from_xml_3d,
    'translation3d': GNTranslation.from_xml_3d,
    'intersection3d': GNIntersection.from_xml_3d
}
geometry_types_3d_core_transforms.update(geometry_types_3d_core_extrusion)
geometry_types_3d_core_transforms.update(geometry_types_3d_core_revolution)

geometry_types_3d_core = {}  # only unique names of types, displayed in add menu
geometry_types_3d_core.update(geometry_types_3d_core_leafs)
geometry_types_3d_core.update(geometry_types_3d_core_containers)
geometry_types_3d_core.update(geometry_types_3d_core_transforms)

geometry_types_3d = {   # all names: geometry_types_2d_core + aliases

    # leafs:
    'block3d': GNBlock.from_xml_3d,
    'circle3d': GNCircle.from_xml_3d,

    # containers:
    'container3d': GNAlignContainer.from_xml_3d,

    # transforms:
}
geometry_types_3d.update(geometry_types_3d_core)


geometry_types_other = {
    'again': GNAgain.from_xml,
    'copy': GNCopy.from_xml
}


geometry_types_geometries_core = {
    'cartesian2d': GNCartesian.from_xml_2d,
    'cartesian3d': GNCartesian.from_xml_3d,
    'cylindrical': GNCylindrical.from_xml_2d
}

geometry_types_geometries = {   # with alternative names
    'cylindrical2d': GNCylindrical.from_xml_2d
}
geometry_types_geometries.update(geometry_types_geometries_core)
