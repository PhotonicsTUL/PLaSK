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
#include "prism.hpp"
#include "reader.hpp"

#include "../manager.hpp"

#define PLASK_PRISM_NAME "prism"

namespace plask {

const char* Prism::NAME = PLASK_PRISM_NAME;

std::string Prism::getTypeName() const { return NAME; }

Prism::Prism(const Prism::Vec2& p0, const Prism::Vec2& p1, double height, const shared_ptr<Material>& material)
    : BaseClass(material), p0(p0), p1(p1), height(height) {}

Prism::Prism(const Prism::Vec2& p0,
             const Prism::Vec2& p1,
             double height,
             shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), p0(p0), p1(p1), height(height) {}

Box3D Prism::getBoundingBox() const {
    return Box3D(std::min(std::min(p0.c0, p1.c0), 0.0), std::min(std::min(p0.c1, p1.c1), 0.0), 0.,
                 std::max(std::max(p0.c0, p1.c0), 0.0), std::max(std::max(p0.c1, p1.c1), 0.0), height);
}

inline static double sign(const Vec<3, double>& p1, const Vec<2, double>& p2, const Vec<2, double>& p3) {
    return (p1.c0 - p3.c0) * (p2.c1 - p3.c1) - (p2.c0 - p3.c0) * (p1.c1 - p3.c1);
}

// Like sign, but with p3 = (0, 0)
inline static double sign0(const Vec<3, double>& p1, const Vec<2, double>& p2) {
    return (p1.c0) * (p2.c1) - (p2.c0) * (p1.c1);
}

bool Prism::contains(const Prism::DVec& p) const {
    if (p.c2 < 0 || p.c2 > height) return false;
    // algorithm comes from:
    // http://stackoverflow.com/questions/2049582/how-to-determine-a-point-in-a-triangle
    // with: v1 -> p0, v2 -> p1, v3 -> (0, 0)
    // maybe barycentric method would be better?
    bool b1 = sign(p, p0, p1) < 0.0;
    bool b2 = sign0(p, p1) < 0.0;
    return (b1 == b2) && (b2 == (sign(p, Primitive<2>::ZERO_VEC, p0) < 0.0));
}

void Prism::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes)
        .attr("a" + axes.getNameForLong(), p0.tran())
        .attr("a" + axes.getNameForTran(), p0.vert())
        .attr("b" + axes.getNameForLong(), p1.tran())
        .attr("b" + axes.getNameForTran(), p1.vert())
        .attr("height", height);
}

void Prism::addPointsAlongToSet(std::set<double>& points,
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

        double x[3] = {0., p0[int(direction)], p1[int(direction)]};

        // Sort x
        if (x[2] < x[0]) std::swap(x[0], x[2]);
        if (x[1] > x[2])
            std::swap(x[1], x[2]);
        else if (x[1] < x[0])
            std::swap(x[1], x[0]);

        for (int i = 0; i < 3; ++i) points.insert(x[i]);
        double dx02 = x[2] - x[0];
        if (dx02 == 0) return;

        for (int i = 0; i < 2; ++i) {
            double dx = x[i + 1] - x[i];
            unsigned maxsteps = max_steps * (dx / dx02);
            unsigned steps = min(unsigned(dx / min_step_size), maxsteps);
            double step = dx / steps;
            for (unsigned j = 1; j < steps; ++j) points.insert(x[i] + j * step);
        }
    }
}

void Prism::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                 unsigned max_steps,
                                 double min_step_size) const {
    if (!materialProvider->isUniform(Primitive<3>::DIRECTION_LONG))
        throw NotImplemented("prismatic mesh for prisms non-uniform in longitudinal direction");
    if (!materialProvider->isUniform(Primitive<3>::DIRECTION_TRAN))
        throw NotImplemented("prismatic mesh for prisms non-uniform in transverse direction");
    std::set<double> vert;
    addPointsAlongToSet(vert, Primitive<3>::DIRECTION_VERT, max_steps, min_step_size);
    typedef typename GeometryObjectD<3>::LineSegment Segment;
    double pv = 0.;
    for (double v : vert) {
        segments.insert(Segment(DVec(0., 0., v), DVec(p0[0], p0[1], v)));
        segments.insert(Segment(DVec(0., 0., v), DVec(p1[0], p1[1], v)));
        segments.insert(Segment(DVec(p0[0], p0[1], v), DVec(p1[0], p1[1], v)));
        if (v != 0.) {
            segments.insert(Segment(DVec(0., 0., pv), DVec(0., 0., v)));
            segments.insert(Segment(DVec(p0[0], p0[1], pv), DVec(p0[0], p0[1], v)));
            segments.insert(Segment(DVec(p1[0], p1[1], pv), DVec(p1[0], p1[1], v)));
        }
        pv = v;
    }
}

shared_ptr<GeometryObject> read_prism(GeometryReader& reader) {
    shared_ptr<Prism> prism(new Prism());
    if (reader.manager.draft) {
        prism->p0.c0 = reader.source.getAttribute("a" + reader.getAxisLongName(), 0.0);
        prism->p0.c1 = reader.source.getAttribute("a" + reader.getAxisTranName(), 0.0);
        prism->p1.c0 = reader.source.getAttribute("b" + reader.getAxisLongName(), 0.0);
        prism->p1.c1 = reader.source.getAttribute("b" + reader.getAxisTranName(), 0.0);
        prism->height = reader.source.getAttribute("height", 0.0);
    } else {
        prism->p0.c0 = reader.source.requireAttribute<double>("a" + reader.getAxisLongName());
        prism->p0.c1 = reader.source.requireAttribute<double>("a" + reader.getAxisTranName());
        prism->p1.c0 = reader.source.requireAttribute<double>("b" + reader.getAxisLongName());
        prism->p1.c1 = reader.source.requireAttribute<double>("b" + reader.getAxisTranName());
        prism->height = reader.source.requireAttribute<double>("height");
    }
    prism->readMaterial(reader);
    reader.source.requireTagEnd();
    return prism;
}

static GeometryReader::RegisterObjectReader prism_reader(PLASK_PRISM_NAME, read_prism);

}  // namespace plask
