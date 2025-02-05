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
#include "elliptic_cylinder.hpp"
#include "../manager.hpp"
#include "reader.hpp"

#define PLASK_ELLIPTIC_CYLINDER_NAME "elliptic-cylinder"

namespace plask {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const char* EllipticCylinder::NAME = PLASK_ELLIPTIC_CYLINDER_NAME;

EllipticCylinder::EllipticCylinder(double radius0,
                                   double radius1,
                                   double angle,
                                   double height,
                                   const shared_ptr<Material>& material)
    : GeometryObjectLeaf<3>(material),
      radius0(std::max(radius0, 0.)),
      radius1(std::max(radius1, 0.)),
      sina(sin(angle)),
      cosa(cos(angle)),
      height(std::max(height, 0.)) {}

EllipticCylinder::EllipticCylinder(double radius0,
                                   double radius1,
                                   double angle,
                                   double height,
                                   const shared_ptr<MaterialsDB::MixedCompositionFactory>& materialTopBottom)
    : GeometryObjectLeaf<3>(materialTopBottom),
      radius0(std::max(radius0, 0.)),
      radius1(std::max(radius1, 0.)),
      sina(sin(angle)),
      cosa(cos(angle)),
      height(std::max(height, 0.)) {}

EllipticCylinder::EllipticCylinder(double radius0, double radius1, double height, const shared_ptr<Material>& material)
    : GeometryObjectLeaf<3>(material),
      radius0(std::max(radius0, 0.)),
      radius1(std::max(radius1, 0.)),
      sina(0.),
      cosa(1.),
      height(std::max(height, 0.)) {}

EllipticCylinder::EllipticCylinder(double radius0,
                                   double radius1,
                                   double height,
                                   const shared_ptr<MaterialsDB::MixedCompositionFactory>& materialTopBottom)
    : GeometryObjectLeaf<3>(materialTopBottom),
      radius0(std::max(radius0, 0.)),
      radius1(std::max(radius1, 0.)),
      sina(0.),
      cosa(1.),
      height(std::max(height, 0.)) {}

EllipticCylinder::EllipticCylinder(const EllipticCylinder& src)
    : GeometryObjectLeaf<3>(src), radius0(src.radius0), radius1(src.radius1), sina(src.sina), cosa(src.cosa), height(src.height) {}

EllipticCylinder::Box EllipticCylinder::getBoundingBox() const {
    double c2 = cosa * cosa, s2 = sina * sina;
    double a2 = radius0 * radius0, b2 = radius1 * radius1;
    double x = sqrt(c2 * a2 + s2 * b2), y = sqrt(s2 * a2 + c2 * b2);
    return Box(DVec(-x, -y, 0), DVec(x, y, height));
}

bool EllipticCylinder::contains(const EllipticCylinder::DVec& p) const {
    if (0.0 > p.vert() || p.vert() > height) return false;
    DVec p1 = invT(p);
    double x = p1.c0 / radius0, y = p1.c1 / radius1;
    return x * x + y * y <= 1.;
}

void EllipticCylinder::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectLeaf<3>::writeXMLAttr(dest_xml_object, axes);
    XMLWriter::Element& w = materialProvider->writeXML(dest_xml_object, axes);
    w.attr("radius0", radius0).attr("radius1", radius1).attr("height", height);
    if (sina != 0.) w.attr("angle", getAngle());
}

void EllipticCylinder::addPointsAlongToSet(std::set<double>& points,
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
        auto bbox = getBoundingBox();
        double b0, b1;
        if (direction == Primitive<3>::DIRECTION_LONG) {
            b0 = bbox.lower.lon();
            b1 = bbox.upper.lon();
        } else {
            b0 = bbox.lower.tran();
            b1 = bbox.upper.tran();
        }
        double d = b1 - b0;
        if (this->max_steps) max_steps = this->max_steps;
        if (this->min_step_size) min_step_size = this->min_step_size;
        unsigned steps = min(unsigned(d / min_step_size), max_steps);
        double step = d / steps;
        for (unsigned i = 0; i <= steps; ++i) points.insert(b0 + i * step);
    }
}

void EllipticCylinder::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                            unsigned max_steps,
                                            double min_step_size) const {
    typedef typename GeometryObjectD<3>::LineSegment Segment;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    unsigned steps = min(unsigned(M_PI * min(radius0, radius1) / min_step_size), max_steps);
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
        double x0 = radius0, y0 = 0;
        if (z1 != 0.) {
            segments.insert(Segment(TVec(-x0, -y0, z0), TVec(-x0, -y0, z1)));
            segments.insert(Segment(TVec(x0, -y0, z0), TVec(x0, -y0, z1)));
            segments.insert(Segment(TVec(-x0, y0, z0), TVec(-x0, y0, z1)));
            segments.insert(Segment(TVec(x0, y0, z0), TVec(x0, y0, z1)));
        }
        for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
            double phi = dphi * i;
            double x1 = radius0 * cos(phi), y1 = radius0 * sin(phi);
            segments.insert(Segment(TVec(-x0, -y0, z1), TVec(-x1, -y1, z1)));
            segments.insert(Segment(TVec(x0, -y0, z1), TVec(x1, -y1, z1)));
            segments.insert(Segment(TVec(-x0, y0, z1), TVec(-x1, y1, z1)));
            segments.insert(Segment(TVec(x0, y0, z1), TVec(x1, y1, z1)));
            if (z1 != 0.) {
                segments.insert(Segment(TVec(-x1, -y1, z0), TVec(-x1, -y1, z1)));
                segments.insert(Segment(TVec(x1, -y1, z0), TVec(x1, -y1, z1)));
                segments.insert(Segment(TVec(-x1, y1, z0), TVec(-x1, y1, z1)));
                segments.insert(Segment(TVec(x1, y1, z0), TVec(x1, y1, z1)));
            }
            if (x1 >= 0 && !materialProvider->isUniform(Primitive<3>::DIRECTION_LONG)) {
                segments.insert(Segment(TVec(-x1, -y1, z1), TVec(-x1, y1, z1)));
                segments.insert(Segment(TVec(x1, -y1, z1), TVec(x1, y1, z1)));
            }
            if (!materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN)) {
                segments.insert(Segment(TVec(-x1, -y1, z1), TVec(x1, -y1, z1)));
                segments.insert(Segment(TVec(-x1, y1, z1), TVec(x1, y1, z1)));
            }
            x0 = x1;
            y0 = y1;
        }
        z0 = z1;
    }
}

shared_ptr<GeometryObject> read_elliptic_cylinder(GeometryReader& reader) {
    shared_ptr<EllipticCylinder> result =
        reader.manager.draft ? plask::make_shared<EllipticCylinder>(reader.source.getAttribute("radius0", 0.0),
                                                                    reader.source.getAttribute("radius1", 0.0),
                                                                    M_PI / 180. * reader.source.getAttribute("angle", 0.0),
                                                                    reader.source.getAttribute("height", 0.0))
                             : plask::make_shared<EllipticCylinder>(reader.source.requireAttribute<double>("radius0"),
                                                                    reader.source.requireAttribute<double>("radius1"),
                                                                    M_PI / 180. * reader.source.getAttribute("angle", 0.0),
                                                                    reader.source.requireAttribute<double>("height"));
    result->readMaterial(reader);
    reader.source.requireTagEnd();
    return result;
}

static GeometryReader::RegisterObjectReader cylinder_reader(PLASK_ELLIPTIC_CYLINDER_NAME, read_elliptic_cylinder);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace plask
