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

// Add characteristic points information along specified axis to set
// \param[in,out] points ordered set of division points along specified axis
// \param direction axis direction
// \param max_steps maximum number of points to split single leaf
// \param min_step_size minimum distance between divisions for a single leaf
void RotatedCuboid::addPointsAlongToSet(std::set<double>& points,
                                        Primitive<3>::Direction direction,
                                        unsigned max_steps,
                                        double min_step_size) const {
    if (direction == Primitive<3>::DIRECTION_VERT) {
        if (this->materialProvider->isUniform(Primitive<3>::DIRECTION_VERT)) {
            points.insert(0);
            points.insert(size[2]);
        } else {
            if (this->max_steps) max_steps = this->max_steps;
            if (this->min_step_size) min_step_size = this->min_step_size;
            double length = size[2];
            unsigned steps = min(unsigned(length / min_step_size), max_steps);
            double step = length / steps;
            for (unsigned i = 0; i <= steps; ++i) points.insert(i * step);
        }
    } else if ((this->c == 0. || this->s == 0.) && this->materialProvider->isUniform(direction)) {
        points.insert(0);
        points.insert(size[size_t(direction)]);
    } else {
        if (this->max_steps) max_steps = this->max_steps;
        if (this->min_step_size) min_step_size = this->min_step_size;
        DVec hl(trans(size.c0, 0.)), hh(trans(size.c0, size.c1)), lh(trans(0., size.c1));
        const size_t dir = size_t(direction);
        double coords[4] = {0., hl[dir], hh[dir], lh[dir]};
        std::sort(coords, coords + 4);
        double total = coords[3] - coords[0];
        for (size_t i = 0; i < 3; ++i) {
            if (coords[i] != coords[i + 1]) points.insert(coords[i]);
            double len = coords[i + 1] - coords[i];
            double dn = std::round(len / total * max_steps);
            size_t n = size_t(dn);
            if (n > 1) {
                double step = len / dn;
                if (step < min_step_size) {
                    dn = std::round(len / min_step_size);
                    n = size_t(dn);
                    step = len / dn;
                }
                for (size_t j = 1; j < n; ++j) points.insert(coords[i] + j * step);
            }
        }
        points.insert(coords[3]);
    }
}

// Add characteristic points to the set and edges connecting them
// \param max_steps maximum number of points to split single leaf
// \param min_step_size minimum distance between divisions for a single leaf
// \param[in, out] segments set to extend
void RotatedCuboid::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                         unsigned max_steps,
                                         double min_step_size) const {
    if (!(materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN) && materialProvider->isUniform(Primitive<3>::DIRECTION_LONG)))
        throw NotImplemented("Triangular mesh for rotated cuboids non-uniform in lateral directions");

    std::vector<double> ptsz;
    std::set<double> ps;
    addPointsAlongToSet(ps, Primitive<3>::Direction(Primitive<3>::DIRECTION_VERT), max_steps, min_step_size);
    ptsz.reserve(ps.size());
    ptsz.insert(ptsz.end(), ps.begin(), ps.end());
    DVec corners[4] = {Primitive<3>::ZERO_VEC, trans(size.c0, 0.), trans(size.c0, size.c1), trans(0., size.c1)};
    for (size_t i = 0; i < 4; ++i) {
        DVec p0 = corners[i];
        DVec p1 = corners[(i + 1) % 4];
        segments.insert(typename GeometryObjectD<3>::LineSegment(p0, p1));
        for (size_t j = 1; j < ptsz.size(); ++j) {
            DVec q0 = p0;
            double z = ptsz[j];
            p0[size_t(Primitive<3>::DIRECTION_VERT)] = z;
            p1[size_t(Primitive<3>::DIRECTION_VERT)] = z;
            segments.insert(typename GeometryObjectD<3>::LineSegment(q0, p0));
            segments.insert(typename GeometryObjectD<3>::LineSegment(p0, p1));
        }
    }
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
