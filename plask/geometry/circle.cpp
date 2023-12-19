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
#include "circle.hpp"
#include "../math.hpp"
#include "../manager.hpp"
#include "reader.hpp"

#define PLASK_CIRCLE2D_NAME "circle"
#define PLASK_CIRCLE3D_NAME "sphere"

namespace plask {

template <int dim> const char* Circle<dim>::NAME = dim == 2 ? PLASK_CIRCLE2D_NAME : PLASK_CIRCLE3D_NAME;

template <int dim> std::string Circle<dim>::getTypeName() const { return NAME; }

template <int dim>
Circle<dim>::Circle(double radius, const shared_ptr<plask::Material>& material)
    : GeometryObjectLeaf<dim>(material), radius(radius) {
    if (radius < 0.) radius = 0.;
}

template <> typename Circle<2>::Box Circle<2>::getBoundingBox() const {
    return Circle<2>::Box(vec(-radius, -radius), vec(radius, radius));
}

template <> typename Circle<3>::Box Circle<3>::getBoundingBox() const {
    return Circle<3>::Box(vec(-radius, -radius, -radius), vec(radius, radius, radius));
}

template <int dim> bool Circle<dim>::contains(const typename Circle<dim>::DVec& p) const {
    return abs2(p) <= radius * radius;
}

template <int dim> void Circle<dim>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectLeaf<dim>::writeXMLAttr(dest_xml_object, axes);
    this->materialProvider->writeXML(dest_xml_object, axes).attr("radius", this->radius);
}

template <int dim>
void Circle<dim>::addPointsAlongToSet(std::set<double>& points,
                                      Primitive<3>::Direction direction,
                                      unsigned max_steps,
                                      double min_step_size) const {
    assert(int(direction) >= 3 - dim && int(direction) <= 3);
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    unsigned steps = min(unsigned(2. * radius / min_step_size), max_steps);
    double step = 2. * radius / steps;
    for (unsigned i = 0; i <= steps; ++i) points.insert(i * step - radius);
}

template <>
void Circle<2>::addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                                     unsigned max_steps,
                                     double min_step_size) const {
    typedef typename GeometryObjectD<2>::LineSegment Segment;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    if (materialProvider->isUniform(Primitive<3>::DIRECTION_VERT)) {
        unsigned steps = min(unsigned(M_PI * radius / min_step_size), max_steps);
        double dphi = M_PI / steps;
        double x0 = radius, y0 = 0;
        for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
            double phi = dphi * i;
            double x1 = radius * cos(phi), y1 = radius * sin(phi);
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
        // If material is not uniform wertically, we use uniform division in vertical direction
        unsigned steps = min(unsigned(2. * radius / min_step_size), max_steps);
        double step = 2. * radius / steps;
        double radius2 = radius * radius;
        double x0 = sqrt(0.5 * step * radius);  // x0 = r sin(φ/2) = r √[(1–cosφ)/2], cosφ = (r-s)/r = 1 – s/r
        double y0 = sqrt(radius2 - x0 * x0);
        segments.insert(Segment(DVec(0., -radius), DVec(-x0, -y0)));
        segments.insert(Segment(DVec(0., -radius), DVec(x0, -y0)));
        segments.insert(Segment(DVec(0., radius), DVec(-x0, y0)));
        segments.insert(Segment(DVec(0., radius), DVec(x0, y0)));
        if (!materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN)) {
            segments.insert(Segment(DVec(-x0, -y0), DVec(-x0, y0)));
            segments.insert(Segment(DVec(0., -radius), DVec(0., radius)));
            segments.insert(Segment(DVec(x0, -y0), DVec(x0, y0)));
        }
        for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
            double y1 = radius - i * step;
            double x1 = sqrt(radius2 - y1 * y1);
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

template <>
void Circle<3>::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                     unsigned max_steps,
                                     double min_step_size) const {
    if (!materialProvider->isUniform(Primitive<3>::DIRECTION_LONG) ||
        materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN)) {
        throw NotImplemented("triangular mesh for sphere non-uniform in any horizontal direction");
    }
    typedef typename GeometryObjectD<3>::LineSegment Segment;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    // if (materialProvider->isUniform(Primitive<3>::DIRECTION_VERT)) {
    unsigned steps = min(unsigned(M_PI * radius / min_step_size), max_steps);
    double dphi = M_PI / steps;
    double r0 = radius, z0 = 0;
    for (unsigned i = 0; i <= (steps + 1) / 2; ++i) {
        double theta = dphi * i;
        double r1 = radius * cos(theta), z1 = radius * sin(theta);
        double x00 = r0, y00 = 0., x10 = r1, y10 = 0.;
        for (unsigned j = 1; j <= 2 * steps; ++j) {
            double phi = j * dphi;
            double x01 = r0 * cos(phi), y01 = r0 * sin(phi), x11 = r1 * cos(phi), y11 = r1 * sin(phi);
            if (i != 0) {
                segments.insert(Segment(DVec(x01, y01, z0), DVec(x11, y11, z1)));
                segments.insert(Segment(DVec(x01, y01, -z0), DVec(x11, y11, -z1)));
            }
            if (abs(r1) > SMALL) {
                segments.insert(Segment(DVec(x10, y10, z1), DVec(x11, y11, z1)));
                segments.insert(Segment(DVec(x10, y10, -z1), DVec(x11, y11, -z1)));
            }
            x00 = x01, y00 = y01, x10 = x11, y10 = y11;
        }
        r0 = r1;
        z0 = z1;
    }
}

template <int dim> shared_ptr<GeometryObject> read_circle(GeometryReader& reader) {
    shared_ptr<Circle<dim>> circle =
        plask::make_shared<Circle<dim>>(reader.manager.draft ? reader.source.getAttribute("radius", 0.0)
                                                             : reader.source.requireAttribute<double>("radius"));
    circle->readMaterial(reader);
    reader.source.requireTagEnd();
    return circle;
}

template struct PLASK_API Circle<2>;
template struct PLASK_API Circle<3>;

static GeometryReader::RegisterObjectReader circle_reader(PLASK_CIRCLE2D_NAME, read_circle<2>);
static GeometryReader::RegisterObjectReader sphere_reader(PLASK_CIRCLE3D_NAME, read_circle<3>);

}  // namespace plask
