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
#include "ellipse.hpp"
#include "../manager.hpp"
#include "../math.hpp"
#include "reader.hpp"

#define PLASK_ELLIPSE2D_NAME "ellipse"

namespace plask {

const char* Ellipse::NAME = PLASK_ELLIPSE2D_NAME;

std::string Ellipse::getTypeName() const { return NAME; }

Ellipse::Ellipse(double rx, double ry, const shared_ptr<plask::Material>& material)
    : GeometryObjectLeaf(material), radius0(std::max(rx, 0.)), radius1(std::max(ry, 0.)) {}

Ellipse::Ellipse(double rx, double ry, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : GeometryObjectLeaf(materialTopBottom), radius0(std::max(rx, 0.)), radius1(std::max(ry, 0.)) {}

typename Ellipse::Box Ellipse::getBoundingBox() const { return Ellipse::Box(vec(-radius0, -radius1), vec(radius0, radius1)); }

bool Ellipse::contains(const typename Ellipse::DVec& p) const {
    double x = p.c0 / radius0, y = p.c1 / radius1;
    return x * x + y * y <= 1.;
}

void Ellipse::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectLeaf::writeXMLAttr(dest_xml_object, axes);
    this->materialProvider->writeXML(dest_xml_object, axes).attr("radius0", this->radius0).attr("radius1", this->radius1);
}

void Ellipse::addPointsAlongToSet(std::set<double>& points,
                                  Primitive<3>::Direction direction,
                                  unsigned max_steps,
                                  double min_step_size) const {
    assert(int(direction) >= 1 && int(direction) <= 3);
    double radius = (direction == Primitive<3>::DIRECTION_VERT) ? radius1 : radius0;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    unsigned steps = min(unsigned(2. * radius / min_step_size), max_steps);
    double step = 2. * radius / steps;
    for (unsigned i = 0; i <= steps; ++i) points.insert(i * step - radius);
}

void Ellipse::addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                                   unsigned max_steps,
                                   double min_step_size) const {
    typedef typename GeometryObjectD<2>::LineSegment Segment;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    if (materialProvider->isUniform(Primitive<3>::DIRECTION_VERT)) {
        unsigned steps = min(unsigned(M_PI * min(radius0, radius1) / min_step_size), max_steps);
        double dphi = M_PI / steps;
        double x0 = radius0, y0 = 0;
        for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
            double phi = dphi * i;
            double x1 = radius0 * cos(phi), y1 = radius1 * sin(phi);
            segments.insert(Segment(DVec(-x0, -y0), DVec(-x1, -y1)));
            segments.insert(Segment(DVec(x0, -y0), DVec(x1, -y1)));
            segments.insert(Segment(DVec(-x0, y0), DVec(-x1, y1)));
            segments.insert(Segment(DVec(x0, y0), DVec(x1, y1)));
            if (x1 >= 0 && !materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN)) {
                segments.insert(Segment(DVec(-x1, -y1), DVec(-x1, y1)));
                segments.insert(Segment(DVec(x1, -y1), DVec(x1, y1)));
            }
            x0 = x1;
            y0 = y1;
        }
    } else {
        // If material is not uniform vertically, we use uniform division in vertical direction
        unsigned steps = min(unsigned(2. * radius1 / min_step_size), max_steps);
        double step = 2. * radius1 / steps;
        double x0 = sqrt(0.5 * step * radius0);  // x0 = r sin(φ/2) = r √[(1–cosφ)/2], cosφ = (r-s)/r = 1 – s/r
        double xr0 = x0 / radius0;
        double y0 = radius1 * sqrt(1. - xr0 * xr0);
        segments.insert(Segment(DVec(0., -radius1), DVec(-x0, -y0)));
        segments.insert(Segment(DVec(0., -radius1), DVec(x0, -y0)));
        segments.insert(Segment(DVec(0., radius1), DVec(-x0, y0)));
        segments.insert(Segment(DVec(0., radius1), DVec(x0, y0)));
        if (!materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN)) {
            segments.insert(Segment(DVec(-x0, -y0), DVec(-x0, y0)));
            segments.insert(Segment(DVec(0., -radius1), DVec(0., radius1)));
            segments.insert(Segment(DVec(x0, -y0), DVec(x0, y0)));
        }
        for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
            double y1 = radius1 - i * step;
            double yr1 = y1 / radius1;
            double x1 = radius0 * sqrt(1. - yr1 * yr1);
            segments.insert(Segment(DVec(-x1, -y1), DVec(x1, -y1)));
            segments.insert(Segment(DVec(-x1, y1), DVec(x1, y1)));
            segments.insert(Segment(DVec(-x0, -y0), DVec(-x1, -y1)));
            segments.insert(Segment(DVec(x0, -y0), DVec(x1, -y1)));
            segments.insert(Segment(DVec(-x0, y0), DVec(-x1, y1)));
            segments.insert(Segment(DVec(x0, y0), DVec(x1, y1)));
            if (!materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN)) {
                segments.insert(Segment(DVec(x1, -y1), DVec(x1, y1)));
                segments.insert(Segment(DVec(-x1, -y1), DVec(-x1, y1)));
            }
            x0 = x1;
            y0 = y1;
        }
    }
}

shared_ptr<GeometryObject> read_ellipse(GeometryReader& reader) {
    shared_ptr<Ellipse> ellipse =
        reader.manager.draft
            ? plask::make_shared<Ellipse>(reader.source.getAttribute("radius0", 0.0), reader.source.getAttribute("radius1", 0.0))
            : plask::make_shared<Ellipse>(reader.source.requireAttribute<double>("radius0"),
                                          reader.source.requireAttribute<double>("radius1"));
    ellipse->readMaterial(reader);
    reader.source.requireTagEnd();
    return ellipse;
}

static GeometryReader::RegisterObjectReader ellipse_reader(PLASK_ELLIPSE2D_NAME, read_ellipse);

}  // namespace plask
