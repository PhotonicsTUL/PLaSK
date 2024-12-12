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

#define PLASK_TRIANGULAR_PRISM_NAME "triangular-prism"
#define PLASK_PRISM_NAME "prism"

namespace plask {

const char* TriangularPrism::NAME = PLASK_TRIANGULAR_PRISM_NAME;

std::string TriangularPrism::getTypeName() const { return NAME; }

TriangularPrism::TriangularPrism(const TriangularPrism::Vec2& p0, const TriangularPrism::Vec2& p1, double height, const shared_ptr<Material>& material)
    : BaseClass(material), p0(p0), p1(p1), height(height) {}

TriangularPrism::TriangularPrism(const TriangularPrism::Vec2& p0,
             const TriangularPrism::Vec2& p1,
             double height,
             shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), p0(p0), p1(p1), height(height) {}

Box3D TriangularPrism::getBoundingBox() const {
    return Box3D(std::min(std::min(p0.c0, p1.c0), 0.0), std::min(std::min(p0.c1, p1.c1), 0.0), 0.,
                 std::max(std::max(p0.c0, p1.c0), 0.0), std::max(std::max(p0.c1, p1.c1), 0.0), height);
}

inline static double sign(const Vec<3, double>& p1, const LateralVec<double>& p2, const LateralVec<double>& p3) {
    return (p1.c0 - p3.c0) * (p2.c1 - p3.c1) - (p2.c0 - p3.c0) * (p1.c1 - p3.c1);
}

// Like sign, but with p3 = (0, 0)
inline static double sign0(const Vec<3, double>& p1, const LateralVec<double>& p2) {
    return (p1.c0) * (p2.c1) - (p2.c0) * (p1.c1);
}

bool TriangularPrism::contains(const TriangularPrism::DVec& p) const {
    if (p.c2 < 0 || p.c2 > height) return false;
    // algorithm comes from:
    // http://stackoverflow.com/questions/2049582/how-to-determine-a-point-in-a-triangle
    // with: v1 -> p0, v2 -> p1, v3 -> (0, 0)
    // maybe barycentric method would be better?
    bool b1 = sign(p, p0, p1) < 0.0;
    bool b2 = sign0(p, p1) < 0.0;
    return (b1 == b2) && (b2 == (sign(p, Primitive<2>::ZERO_VEC, p0) < 0.0));
}

void TriangularPrism::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes)
        .attr("a" + axes.getNameForLong(), p0.tran())
        .attr("a" + axes.getNameForTran(), p0.vert())
        .attr("b" + axes.getNameForLong(), p1.tran())
        .attr("b" + axes.getNameForTran(), p1.vert())
        .attr("height", height);
}

void TriangularPrism::addPointsAlongToSet(std::set<double>& points,
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

void TriangularPrism::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
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

shared_ptr<GeometryObject> read_triangular_prism(GeometryReader& reader) {
    shared_ptr<TriangularPrism> prism(new TriangularPrism());
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

static GeometryReader::RegisterObjectReader prism_reader(PLASK_TRIANGULAR_PRISM_NAME, read_triangular_prism);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const char* Prism::NAME = PLASK_PRISM_NAME;

std::string Prism::getTypeName() const { return NAME; }

Prism::Prism(double height, const std::vector<LateralVec<double>>& vertices, const shared_ptr<Material>& material)
    : BaseClass(material), height(height), vertices(vertices) {}

Prism::Prism(double height, std::vector<LateralVec<double>>&& vertices, const shared_ptr<Material>&& material)
    : BaseClass(material), height(height), vertices(std::move(vertices)) {}

Prism::Prism(double height, std::initializer_list<LateralVec<double>> vertices, const shared_ptr<Material>& material)
    : BaseClass(material), height(height), vertices(vertices) {}

Prism::Prism(double height,
                     const std::vector<LateralVec<double>>& vertices,
                     shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), height(height), vertices(vertices) {}

Prism::Prism(double height,
                     std::vector<LateralVec<double>>&& vertices,
                     shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), height(height), vertices(std::move(vertices)) {}

Prism::Prism(double height,
                     std::initializer_list<LateralVec<double>> vertices,
                     shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), height(height), vertices(vertices) {}

void Prism::validate() const {
    if (vertices.size() < 3) {
        throw GeometryException("polygon has less than 3 vertices");
    }
    // if (!checkSegments()) {
    //     throw GeometryException("polygon has intersecting segments");
    // }
}

// bool Prism::checkSegments() const {
//     for (size_t i = 0; i < vertices.size(); ++i) {
//         LateralVec<double> a1 = vertices[i];
//         LateralVec<double> b1 = vertices[(i + 1) % vertices.size()];
//         double min1x = std::min(a1.c0, b1.c0), max1x = std::max(a1.c0, b1.c0);
//         double min1y = std::min(a1.c1, b1.c1), max1y = std::max(a1.c1, b1.c1);
//         double d1x = b1.c0 - a1.c0;
//         double d1y = b1.c1 - a1.c1;
//         double det1 = a1.c0 * b1.c1 - a1.c1 * b1.c0;
//         for (size_t j = i + 2; j < vertices.size() - (i ? 0 : 1); ++j) {
//             LateralVec<double> a2 = vertices[j];
//             LateralVec<double> b2 = vertices[(j + 1) % vertices.size()];
//             double min2x = std::min(a2.c0, b2.c0), max2x = std::max(a2.c0, b2.c0);
//             double min2y = std::min(a2.c1, b2.c1), max2y = std::max(a2.c1, b2.c1);
//             if (max2x < min1x || max1x < min2x || max2y < min1y || max1y < min2y) continue;
//             double d2x = b2.c0 - a2.c0;
//             double d2y = b2.c1 - a2.c1;
//             double det = d1x * d2y - d2x * d1y;
//             double det2 = a2.c0 * b2.c1 - a2.c1 * b2.c0;
//             if (det == 0) continue;
//             double x = (d1x * det2 - d2x * det1) / det;
//             double y = (d1y * det2 - d2y * det1) / det;
//             if (x >= min1x && x <= max1x && x >= min2x && x <= max2x && y >= min1y && y <= max1y && y >= min2y && y <= max2y)
//                 return false;
//         }
//     }
//     return true;
// }

Prism::Box Prism::getBoundingBox() const {
    if (vertices.empty()) return Box(DVec(0, 0, 0), DVec(0, 0, 0));
    double min_x = vertices[0].c0;
    double max_x = vertices[0].c0;
    double min_y = vertices[0].c1;
    double max_y = vertices[0].c1;
    for (const LateralVec<double>& v : vertices) {
        min_x = std::min(min_x, v.c0);
        max_x = std::max(max_x, v.c0);
        min_y = std::min(min_y, v.c1);
        max_y = std::max(max_y, v.c1);
    }
    return Box(DVec(min_x, min_y, 0), DVec(max_x, max_y, height));
}

bool Prism::contains(const DVec& p) const {
    if (vertices.size() < 3) return false;
    if (p.c2 < 0 || p.c2 > height) return false;
    int n = vertices.size();
    int i, j;
    int c = 0;
    for (i = 0, j = n - 1; i < n; j = i++) {
        if (((vertices[i].c1 > p.c1) != (vertices[j].c1 > p.c1)) &&
            (p.c0 <
             (vertices[j].c0 - vertices[i].c0) * (p.c1 - vertices[i].c1) / (vertices[j].c1 - vertices[i].c1) + vertices[i].c0))
            c += (vertices[i].c1 > vertices[j].c1) ? 1 : -1;
    }
    return c;
}

void Prism::addPointsAlongToSet(std::set<double>& points,
                                    Primitive<3>::Direction direction,
                                    unsigned max_steps,
                                    double min_step_size) const {
    if (direction == Primitive<3>::Direction::DIRECTION_VERT) {
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
        return;
    }

    if (vertices.size() < 3) return;
    std::set<double> vert;
    for (const LateralVec<double>& v : vertices) {
        vert.insert(v[int(direction)]);
    }
    for (std::set<double>::const_iterator b = vert.begin(), a = b++; b != vert.end(); ++a, ++b) {
        double d = *b - *a;
        unsigned steps = std::max(1u, static_cast<unsigned>(d / min_step_size));
        steps = std::min(steps, max_steps);
        double step = d / steps;
        for (unsigned i = 0; i <= steps; ++i) points.insert(*a + i * step);
    }
}

void Prism::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                     unsigned max_steps,
                                     double min_step_size) const {
    if (vertices.size() < 3) return;
    typedef typename GeometryObjectD<3>::LineSegment Segment;
    for (size_t i = 0; i < vertices.size(); ++i) {
        LateralVec<double> a = vertices[i];
        LateralVec<double> b = vertices[(i + 1) % vertices.size()];
        LateralVec<double> ab = b - a;
        double d = std::sqrt(dot(ab, ab));
        unsigned steps = std::max(1u, static_cast<unsigned>(d / min_step_size));
        steps = std::min(steps, max_steps);
        LateralVec<double> p0 = a;
        for (unsigned j = 1; j <= steps; ++j) {
            segments.insert(Segment(DVec(p0.c0, p0.c1, 0), DVec(p0.c0, p0.c1, height)));
            double t = static_cast<double>(j) / steps;
            LateralVec<double> p = a * (1 - t) + b * t;
            segments.insert(Segment(DVec(p0.c0, p0.c1, 0), DVec(p.c0, p.c1, 0)));
            segments.insert(Segment(DVec(p0.c0, p0.c1, height), DVec(p.c0, p.c1, height)));
        }
    }
}

void Prism::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes).attr("height", height);
    if (vertices.empty()) return;
    std::string vertices_str;
    const char* sep = "";
    for (const LateralVec<double>& v : vertices) {
        vertices_str += sep;
        vertices_str += str(v.c0) + " " + str(v.c1);
        sep = "; ";
    }
    dest_xml_object.writeText(vertices_str);
}

shared_ptr<GeometryObject> read_prism(GeometryReader& reader) {

    // DEPRECATED: use TriangularPrism instead
    if (reader.source.hasAttribute("a" + reader.getAxisLongName()) ||
        reader.source.hasAttribute("a" + reader.getAxisTranName()) ||
        reader.source.hasAttribute("b" + reader.getAxisLongName()) ||
        reader.source.hasAttribute("b" + reader.getAxisTranName())) {
        writelog(LOG_WARNING, "<prism> with vertices a and b is deprecated, use <triangular-prism> instead");
        return read_triangular_prism(reader);
    }

    shared_ptr<Prism> prism = make_shared<Prism>();
    prism->readMaterial(reader);
    if (reader.manager.draft)
        prism->height = reader.source.getAttribute("height", 0.0);
    else
        prism->height = reader.source.requireAttribute<double>("height");

    std::string vertex_spec = reader.source.requireTextInCurrentTag();
    if (reader.source.attributeFilter) vertex_spec = reader.source.attributeFilter(vertex_spec);
    std::vector<LateralVec<double>> vertices;
    boost::tokenizer<boost::char_separator<char>> tokens(vertex_spec, boost::char_separator<char>(" \t\n\r", ";"));
    int vi = 0;
    for (const std::string& t : tokens) {
        if (t == ";") {  // end of point or segment
            if (vi != 2) throw Exception("each vertex must have two coordinates");
            vi = 0;
        } else {  // end of point coordinate
            if (vi == 2) throw Exception("end of vertex (\";\") was expected, but got \"{0}\"", t);
            if (vi == 0) vertices.emplace_back();
            try {
                vertices.back()[vi++] = boost::lexical_cast<double>(t);
            } catch (const boost::bad_lexical_cast&) {
                throw Exception("bad vertex coordinate: {0}", t);
            }
        }
    }
    prism->vertices = std::move(vertices);
    if (!reader.manager.draft) prism->validate();
    return prism;
}

static GeometryReader::RegisterObjectReader polygon_reader(PLASK_PRISM_NAME, read_prism);

}  // namespace plask
