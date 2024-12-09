/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2024 Lodz University of Technology
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

#include "polygon.hpp"

namespace plask {

const char* Polygon::NAME = "polygon";

std::string Polygon::getTypeName() const { return NAME; }

Polygon::Polygon(const std::vector<Vec<2>>& vertices, const shared_ptr<Material>& material)
    : BaseClass(material), vertices(vertices) {}

Polygon::Polygon(std::vector<Vec<2>>&& vertices, const shared_ptr<Material>&& material)
    : BaseClass(material), vertices(std::move(vertices)) {}

Polygon::Polygon(std::initializer_list<Vec<2>> vertices, const shared_ptr<Material>& material)
    : BaseClass(material), vertices(vertices) {}

Polygon::Polygon(const std::vector<Vec<2>>& vertices, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), vertices(vertices) {}

Polygon::Polygon(std::vector<Vec<2>>&& vertices, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), vertices(std::move(vertices)) {}

Polygon::Polygon(std::initializer_list<Vec<2>> vertices, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), vertices(vertices) {}

void Polygon::validate() const {
    if (vertices.size() < 3) {
        throw GeometryException("polygon has less than 3 vertices");
    }
    // if (!checkSegments()) {
    //     throw GeometryException("polygon has intersecting segments");
    // }
}

// bool Polygon::checkSegments() const {
//     for (size_t i = 0; i < vertices.size(); ++i) {
//         Vec<2> a1 = vertices[i];
//         Vec<2> b1 = vertices[(i + 1) % vertices.size()];
//         double min1x = std::min(a1.c0, b1.c0), max1x = std::max(a1.c0, b1.c0);
//         double min1y = std::min(a1.c1, b1.c1), max1y = std::max(a1.c1, b1.c1);
//         double d1x = b1.c0 - a1.c0;
//         double d1y = b1.c1 - a1.c1;
//         double det1 = a1.c0 * b1.c1 - a1.c1 * b1.c0;
//         for (size_t j = i + 2; j < vertices.size() - (i ? 0 : 1); ++j) {
//             Vec<2> a2 = vertices[j];
//             Vec<2> b2 = vertices[(j + 1) % vertices.size()];
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

Polygon::Box Polygon::getBoundingBox() const {
    if (vertices.empty()) return Box(Vec<2>(0, 0), Vec<2>(0, 0));
    double min_x = vertices[0].c0;
    double max_x = vertices[0].c0;
    double min_y = vertices[0].c1;
    double max_y = vertices[0].c1;
    for (const Vec<2>& v : vertices) {
        min_x = std::min(min_x, v.c0);
        max_x = std::max(max_x, v.c0);
        min_y = std::min(min_y, v.c1);
        max_y = std::max(max_y, v.c1);
    }
    return Box(Vec<2>(min_x, min_y), Vec<2>(max_x, max_y));
}

bool Polygon::contains(const DVec& p) const {
    if (vertices.size() < 3) return false;
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

void Polygon::addPointsAlongToSet(std::set<double>& points,
                                  Primitive<3>::Direction direction,
                                  unsigned max_steps,
                                  double min_step_size) const {
    // TODO: make it more clever reducing the number of points to absolute minimum
    if (vertices.size() < 3) return;
    for (size_t i = 0; i < vertices.size(); ++i) {
        Vec<2> a = vertices[i];
        Vec<2> b = vertices[(i + 1) % vertices.size()];
        Vec<2> ab = b - a;
        double d = std::sqrt(dot(ab, ab));
        unsigned steps = std::max(1u, static_cast<unsigned>(d / min_step_size));
        steps = std::min(steps, max_steps);
        for (unsigned j = 0; j <= steps; ++j) {
            double t = static_cast<double>(j) / steps;
            Vec<2> p = a * (1 - t) + b * t;
            points.insert(p.c0);
        }
    }
}

void Polygon::addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                                   unsigned max_steps,
                                   double min_step_size) const {
    if (vertices.size() < 3) return;
    for (size_t i = 0; i < vertices.size(); ++i) {
        Vec<2> a = vertices[i];
        Vec<2> b = vertices[(i + 1) % vertices.size()];
        Vec<2> ab = b - a;
        double d = std::sqrt(dot(ab, ab));
        unsigned steps = std::max(1u, static_cast<unsigned>(d / min_step_size));
        steps = std::min(steps, max_steps);
        Vec<2> p0 = a;
        for (unsigned j = 1; j <= steps; ++j) {
            double t = static_cast<double>(j) / steps;
            Vec<2> p = a * (1 - t) + b * t;
            segments.insert({p0, p});
            p0 = p;
        }
    }
}

void Polygon::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes);
    if (vertices.empty()) return;
    std::string vertices_str;
    const char* sep = "";
    for (const Vec<2>& v : vertices) {
        vertices_str += sep;
        vertices_str += str(v.c0) + " " + str(v.c1);
        sep = "; ";
    }
    dest_xml_object.writeText(vertices_str);
}

shared_ptr<GeometryObject> readPolygon(GeometryReader& reader) {
    shared_ptr<Polygon> polygon = make_shared<Polygon>();
    polygon->readMaterial(reader);
    std::string vertex_spec = reader.source.requireTextInCurrentTag();
    if (reader.source.attributeFilter) vertex_spec = reader.source.attributeFilter(vertex_spec);
    std::vector<Vec<2>> vertices;
    boost::tokenizer<boost::char_separator<char>> tokens(vertex_spec, boost::char_separator<char>(" \t\n\r", ";"));
    int vi = 0;
    for (const std::string& t : tokens) {
        if (t == ";") {  // end of point or segment
            if (vi != 2) throw Exception("each vertex must have two coordinates");
            vi = 0;
        } else {  // end of point coordinate
            if (vi == 2)
                throw Exception("end of vertex (\";\") was expected, but got \"{0}\"", t);
            if (vi == 0) vertices.emplace_back();
            try {
                vertices.back()[vi++] = boost::lexical_cast<double>(t);
            } catch (const boost::bad_lexical_cast&) {
                throw Exception("bad vertex coordinate: {0}", t);
            }
        }
    }
    polygon->vertices = std::move(vertices);
    if (!reader.manager.draft) polygon->validate();
    return polygon;
}

static GeometryReader::RegisterObjectReader polygon_reader(Polygon::NAME, readPolygon);

}  // namespace plask
