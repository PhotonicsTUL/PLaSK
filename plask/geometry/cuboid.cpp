/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "cuboid.hpp"

#define PLASK_CUBOID_NAME "cuboid"

namespace plask {

const char* RotatedCuboid::NAME = PLASK_CUBOID_NAME;

std::string RotatedCuboid::getTypeName() const { return NAME; }

RotatedCuboid::Box RotatedCuboid::getBoundingBox() const {
    DVec hl(trans(size.c0, 0.)), hh(trans(size.c0, size.c1)), lh(trans(0., size.c1));
    if (c >= 0) {
        if (s >= 0)
            return Box(lh.c0, 0., 0., hl.c0, hh.c1, size.c2);  // I
        else
            return Box(0., hl.c1, 0., hh.c0, lh.c1, size.c2);  // IV
    } else {
        if (s >= 0)
            return Box(hh.c0, lh.c1, 0., 0., hl.c1, size.c2);  // II
        else
            return Box(hl.c0, hh.c1, 0., lh.c0, 0., size.c2);  // III
    }
}

bool RotatedCuboid::contains(const RotatedCuboid::DVec& p) const { return Box(Primitive<3>::ZERO_VEC, size).contains(itrans(p)); }

void RotatedCuboid::addPointsAlongToSet(std::set<double>& points,
                                        Primitive<3>::Direction direction,
                                        unsigned max_steps,
                                        double min_step_size) const {
    throw NotImplemented("RotatedCuboid::addPointsAlongToSet");
    // Add characteristic points information along specified axis to set
    // \param[in,out] points ordered set of division points along specified axis
    // \param direction axis direction
    // \param max_steps maximum number of points to split single leaf
    // \param min_step_size minimum distance between divisions for a single leaf
}

void RotatedCuboid::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                         unsigned max_steps,
                                         double min_step_size) const {
    throw NotImplemented("RotatedCuboid::addLineSegmentsToSet");
    // Add characteristic points to the set and edges connecting them
    // \param max_steps maximum number of points to split single leaf
    // \param min_step_size minimum distance between divisions for a single leaf
    // \param[in, out] segments set to extend
}

void RotatedCuboid::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    Block<3>::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr("angle", getAngle());
}

shared_ptr<GeometryObject> read_cuboid(GeometryReader& reader) {
    shared_ptr<Block<3>> block;
    if (reader.source.hasAttribute("angle")) {
        block.reset(new RotatedCuboid(reader.source.requireAttribute<double>("angle")));
    } else {
        block.reset(new Block<3>());
    }
    block->size.lon() = details::readAlternativeAttrs(reader, "d" + reader.getAxisLongName(), "length");
    details::setupBlock2D3D(reader, *block);
    return block;
}

static GeometryReader::RegisterObjectReader cuboid_reader(PLASK_CUBOID_NAME, read_cuboid);

}  // namespace plask
