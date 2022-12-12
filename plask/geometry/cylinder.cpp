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
#include "cylinder.hpp"
#include "../manager.hpp"
#include "reader.hpp"

#define PLASK_CYLINDER_NAME "cylinder"

namespace plask {

const char* Cylinder::NAME = PLASK_CYLINDER_NAME;

Cylinder::Cylinder(double radius, double height, const shared_ptr<Material>& material)
    : GeometryObjectLeaf<3>(material), radius(std::max(radius, 0.)), height(std::max(height, 0.)) {}

Cylinder::Cylinder(double radius, double height, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : GeometryObjectLeaf<3>(materialTopBottom), radius(std::max(radius, 0.)), height(std::max(height, 0.)) {}

Cylinder::Cylinder(const Cylinder& src) : GeometryObjectLeaf<3>(src), radius(src.radius), height(src.height) {}

Cylinder::Box Cylinder::getBoundingBox() const { return Box(vec(-radius, -radius, 0.0), vec(radius, radius, height)); }

bool Cylinder::contains(const Cylinder::DVec& p) const {
    return 0.0 <= p.vert() && p.vert() <= height && std::fma(p.lon(), p.lon(), p.tran() * p.tran()) <= radius * radius;
}

void Cylinder::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectLeaf<3>::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes).attr("radius", radius).attr("height", height);
}

void Cylinder::addPointsAlongToSet(std::set<double>& points,
                                   Primitive<3>::Direction direction,
                                   unsigned max_steps,
                                   double min_step_size) const {
    if (direction == Primitive<3>::DIRECTION_VERT) {
        if (materialProvider->isUniform(Primitive<3>::DIRECTION_VERT)) {
            points.insert(0);
            points.insert(height);
        } else {
            if (this->max_steps) max_steps = this->max_steps;
            if (this->min_step_size) min_step_size = this->min_step_size;
            unsigned steps = min(unsigned(height / min_step_size), max_steps);
            double step = height / steps;
            for (unsigned i = 0; i <= steps; ++i) points.insert(i * step);
        }
    } else {
        if (this->max_steps) max_steps = this->max_steps;
        if (this->min_step_size) min_step_size = this->min_step_size;
        unsigned steps = min(unsigned(2. * radius / min_step_size), max_steps);
        double step = 2. * radius / steps;
        for (unsigned i = 0; i <= steps; ++i) points.insert(i * step - radius);
    }
}

void Cylinder::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                    unsigned max_steps,
                                    double min_step_size) const {
    typedef typename GeometryObjectD<3>::LineSegment Segment;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    unsigned steps = min(unsigned(M_PI * radius / min_step_size), max_steps);
    double dphi = M_PI / steps;

    // Make vertical axis
    std::vector<double> zz;
    if (materialProvider->isUniform(Primitive<3>::DIRECTION_VERT)) {
        zz.reserve(2);
        zz.push_back(0);
        zz.push_back(height);
    } else {
        unsigned zsteps = min(unsigned(height / min_step_size), max_steps);
        double zstep = height / zsteps;
        zz.reserve(zsteps + 1);
        for (unsigned i = 0; i <= zsteps; ++i) zz.push_back(i * zstep);
    }

    double z0;
    for (double z1 : zz) {
        double x0 = radius, y0 = 0;
        if (z1 != 0.) {
            segments.insert(Segment(DVec(-x0, -y0, z0), DVec(-x0, -y0, z1)));
            segments.insert(Segment(DVec(x0, -y0, z0), DVec(x0, -y0, z1)));
            segments.insert(Segment(DVec(-x0, y0, z0), DVec(-x0, y0, z1)));
            segments.insert(Segment(DVec(x0, y0, z0), DVec(x0, y0, z1)));
        }
        for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
            double phi = dphi * i;
            double x1 = radius * cos(phi), y1 = radius * sin(phi);
            segments.insert(Segment(DVec(-x0, -y0, z1), DVec(-x1, -y1, z1)));
            segments.insert(Segment(DVec(x0, -y0, z1), DVec(x1, -y1, z1)));
            segments.insert(Segment(DVec(-x0, y0, z1), DVec(-x1, y1, z1)));
            segments.insert(Segment(DVec(x0, y0, z1), DVec(x1, y1, z1)));
            if (z1 != 0.) {
                segments.insert(Segment(DVec(-x1, -y1, z0), DVec(-x1, -y1, z1)));
                segments.insert(Segment(DVec(x1, -y1, z0), DVec(x1, -y1, z1)));
                segments.insert(Segment(DVec(-x1, y1, z0), DVec(-x1, y1, z1)));
                segments.insert(Segment(DVec(x1, y1, z0), DVec(x1, y1, z1)));
            }
            if (x1 >= 0 && !materialProvider->isUniform(Primitive<3>::DIRECTION_LONG)) {
                segments.insert(Segment(DVec(-x1, -y1, z1), DVec(-x1, y1, z1)));
                segments.insert(Segment(DVec(x1, -y1, z1), DVec(x1, y1, z1)));
            }
            if (!materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN)) {
                segments.insert(Segment(DVec(-x1, -y1, z1), DVec(x1, -y1, z1)));
                segments.insert(Segment(DVec(-x1, y1, z1), DVec(x1, y1, z1)));
            }
            x0 = x1;
            y0 = y1;
        }
        z0 = z1;
    }
}

shared_ptr<GeometryObject> read_cylinder(GeometryReader& reader) {
    shared_ptr<Cylinder> result(new Cylinder(reader.manager.draft ? reader.source.getAttribute("radius", 0.0)
                                                                  : reader.source.requireAttribute<double>("radius"),
                                             reader.manager.draft ? reader.source.getAttribute("height", 0.0)
                                                                  : reader.source.requireAttribute<double>("height")));
    result->readMaterial(reader);
    reader.source.requireTagEnd();
    return result;
}

static GeometryReader::RegisterObjectReader cylinder_reader(PLASK_CYLINDER_NAME, read_cylinder);

}  // namespace plask
