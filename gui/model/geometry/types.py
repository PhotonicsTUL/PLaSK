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
from .container import GNStack, GNAlignContainer, GNShelf
from .transform import GNClip, GNExtrusion, GNFlip, GNMirror, GNRevolution, GNTranslation
from .copy import GNCopy, GNAgain
from .geometry import GNCartesian, GNCylindrical


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
    'shelf2d': GNShelf.from_xml_2d,

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
    'again': GNAgain.from_xml,
    'copy': GNCopy.from_xml
}

geometry_types_geometries = {
    'cartesian2d': GNCartesian.from_xml_2d,
    'cartesian3d': GNCartesian.from_xml_3d,
    'cylindrical': GNCylindrical.from_xml_2d,
    'cylindrical2d': GNCylindrical.from_xml_2d
}

