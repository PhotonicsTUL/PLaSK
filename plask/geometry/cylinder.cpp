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
#define PLASK_HOLLOW_CYLINDER_NAME "tube"

namespace plask {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    double z0 = 0.;
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
    shared_ptr<Cylinder> result(new Cylinder(
        reader.manager.draft ? reader.source.getAttribute("radius", 0.0) : reader.source.requireAttribute<double>("radius"),
        reader.manager.draft ? reader.source.getAttribute("height", 0.0) : reader.source.requireAttribute<double>("height")));
    result->readMaterial(reader);
    reader.source.requireTagEnd();
    return result;
}

static GeometryReader::RegisterObjectReader cylinder_reader(PLASK_CYLINDER_NAME, read_cylinder);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const char* HollowCylinder::NAME = PLASK_CYLINDER_NAME;

HollowCylinder::HollowCylinder(double inner_radius, double outer_radius, double height, const shared_ptr<Material>& material)
    : GeometryObjectLeaf<3>(material),
      inner_radius(std::max(inner_radius, 0.)),
      outer_radius(std::max(outer_radius, 0.)),
      height(std::max(height, 0.)) {
    if (inner_radius > outer_radius) throw BadInput("Tube", "Inner radius must be less than outer radius");
}

HollowCylinder::HollowCylinder(double inner_radius,
                               double outer_radius,
                               double height,
                               shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : GeometryObjectLeaf<3>(materialTopBottom),
      inner_radius(std::max(inner_radius, 0.)),
      outer_radius(std::max(outer_radius, 0.)),
      height(std::max(height, 0.)) {
    if (inner_radius > outer_radius) throw BadInput("Tube", "Inner radius must be less than outer radius");
}

HollowCylinder::HollowCylinder(const HollowCylinder& src)
    : GeometryObjectLeaf<3>(src), inner_radius(src.inner_radius), outer_radius(src.outer_radius), height(src.height) {}

HollowCylinder::Box HollowCylinder::getBoundingBox() const {
    return Box(vec(-outer_radius, -outer_radius, 0.0), vec(outer_radius, outer_radius, height));
}

bool HollowCylinder::contains(const HollowCylinder::DVec& p) const {
    return 0.0 <= p.vert() && p.vert() <= height &&
           std::fma(p.lon(), p.lon(), p.tran() * p.tran()) <= outer_radius * outer_radius &&
           std::fma(p.lon(), p.lon(), p.tran() * p.tran()) >= inner_radius * inner_radius;
}

void HollowCylinder::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectLeaf<3>::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes)
        .attr("inner-radius", inner_radius)
        .attr("outer-radius", outer_radius)
        .attr("height", height);
}

void HollowCylinder::addPointsAlongToSet(std::set<double>& points,
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
        int max_steps_2 = int(round(max_steps * inner_radius / outer_radius));
        int max_steps_1 = max_steps - max_steps_2 / 2;
        int steps1 = min(int((outer_radius - inner_radius) / min_step_size), max_steps_1);
        int steps2 = min(int(2 * inner_radius / min_step_size), max_steps_2);
        double step1 = (outer_radius - inner_radius) / steps1;
        double step2 = 2. * inner_radius / steps2;
        for (int i = 0; i < steps1; ++i) points.insert(i * step1 - outer_radius);
        for (int i = 0; i < steps2; ++i) points.insert(i * step2 - inner_radius);
        for (int i = steps1; i >= 0; ++i) points.insert(outer_radius - i * step1);
    }
}

void HollowCylinder::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                          unsigned max_steps,
                                          double min_step_size) const {
    typedef typename GeometryObjectD<3>::LineSegment Segment;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    unsigned steps = min(unsigned(M_PI * inner_radius / min_step_size), max_steps);
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

    double z0 = 0.;
    for (double z1 : zz) {
        double x0 = outer_radius, y0 = 0;
        if (z1 != 0.) {
            segments.insert(Segment(DVec(-x0, -y0, z0), DVec(-x0, -y0, z1)));
            segments.insert(Segment(DVec(x0, -y0, z0), DVec(x0, -y0, z1)));
            segments.insert(Segment(DVec(-x0, y0, z0), DVec(-x0, y0, z1)));
            segments.insert(Segment(DVec(x0, y0, z0), DVec(x0, y0, z1)));
        }
        for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
            double phi = dphi * i;
            double x1 = outer_radius * cos(phi), y1 = outer_radius * sin(phi);
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

    z0 = 0.;
    for (double z1 : zz) {
        double x0 = inner_radius, y0 = 0;
        if (z1 != 0.) {
            segments.insert(Segment(DVec(-x0, -y0, z0), DVec(-x0, -y0, z1)));
            segments.insert(Segment(DVec(x0, -y0, z0), DVec(x0, -y0, z1)));
            segments.insert(Segment(DVec(-x0, y0, z0), DVec(-x0, y0, z1)));
            segments.insert(Segment(DVec(x0, y0, z0), DVec(x0, y0, z1)));
        }
        for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
            double phi = dphi * i;
            double x1 = inner_radius * cos(phi), y1 = inner_radius * sin(phi);
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

shared_ptr<GeometryObject> read_hollow_cylinder(GeometryReader& reader) {
    double inner_radius = reader.manager.draft ? reader.source.getAttribute("inner-radius", 0.0)
                                               : reader.source.requireAttribute<double>("inner-radius");
    double outer_radius = reader.manager.draft ? reader.source.getAttribute("outer-radius", 0.0)
                                               : reader.source.requireAttribute<double>("outer-radius");
    if (reader.manager.draft && inner_radius > outer_radius) inner_radius = outer_radius;
    shared_ptr<HollowCylinder> result(new HollowCylinder(
        inner_radius, outer_radius,
        reader.manager.draft ? reader.source.getAttribute("height", 0.0) : reader.source.requireAttribute<double>("height")));
    result->readMaterial(reader);
    reader.source.requireTagEnd();
    return result;
}

static GeometryReader::RegisterObjectReader hollow_cylinder_reader(PLASK_HOLLOW_CYLINDER_NAME, read_hollow_cylinder);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace plask
