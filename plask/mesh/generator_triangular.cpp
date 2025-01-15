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
#include "generator_triangular.hpp"
#include "triangular2d.hpp"

#include <triangle.h>
using namespace triangle;

namespace plask {

struct TrifreeCaller {
    void operator()(void* ptr) const { trifree(ptr); }
};

struct VecFuzzyCompare {
    bool operator()(const typename Primitive<2>::DVec& a, const typename Primitive<2>::DVec& b) const {
        return Primitive<2>::vecFuzzyCompare(a, b);
    }
};

shared_ptr<MeshD<2>> TriangleGenerator::generate(const shared_ptr<GeometryObjectD<2>>& geometry) {
    typedef typename GeometryObjectD<2>::LineSegment LineSegment;
    typedef typename Primitive<2>::DVec DVec;
    typedef std::map<typename Primitive<2>::DVec, int, VecFuzzyCompare> PointMap;

    std::set<LineSegment> lineSegments = geometry->getLineSegments();

    // Make sets of points and segments removing duplicates
    Box2D bb = geometry->getBoundingBox();

    PointMap pointmap;
    std::set<std::pair<int, int>> segmentsset;
    if (full) {
        pointmap[DVec(bb.left(), bb.bottom())] = 0;
        pointmap[DVec(bb.right(), bb.bottom())] = 1;
        pointmap[DVec(bb.right(), bb.top())] = 2;
        pointmap[DVec(bb.left(), bb.top())] = 3;

        if (pointmap.size() < 4)
            throw BadMesh("TriangularGenerator", "geometry object is too small to create triangular mesh");

        segmentsset.insert(std::make_pair(0, 1));
        segmentsset.insert(std::make_pair(1, 2));
        segmentsset.insert(std::make_pair(2, 3));
        segmentsset.insert(std::make_pair(0, 3));
    }

    size_t n = pointmap.size();
    for (const LineSegment& segment : lineSegments) {
        int iseg[2];
        for (int i = 0; i < 2; ++i) {
            bool ok;
            PointMap::iterator it;
PLASK_NO_CONVERSION_WARNING_BEGIN
            std::tie(it, ok) = pointmap.insert(std::make_pair(segment[i], n));
PLASK_NO_WARNING_END
            if (ok) ++n;
            iseg[i] = it->second;
        }
        if (iseg[0] != iseg[1]) {
            if (iseg[0] > iseg[1]) std::swap(iseg[0], iseg[1]);
            segmentsset.insert(std::make_pair(iseg[0], iseg[1]));
        }
    }

    triangulateio in = {}, out = {};  // are fields are nulled, so we will only fill fields we need

    in.numberofpoints = pointmap.size();
    std::unique_ptr<double[]> in_points(new double[2 * in.numberofpoints]);
    in.pointlist = in_points.get();
    for (auto pi : pointmap) {
        in_points[2 * pi.second] = pi.first.c0;
        in_points[2 * pi.second + 1] = pi.first.c1;
        // std::cerr << format("{}: {:5.3f}, {:5.3f}\n", pi.second, pi.first.c0, pi.first.c1);
    }

    in.numberofsegments = segmentsset.size();
    std::unique_ptr<int[]> in_segments(new int[2 * in.numberofsegments]);
    in.segmentlist = in_segments.get();
    n = 0;
    for (auto s : segmentsset) {
        in_segments[n] = s.first;
        in_segments[n + 1] = s.second;
        n += 2;
        // std::cerr << s.first << "-" << s.second << "\n";
    }

    triangulate(const_cast<char*>(getSwitches().c_str()), &in, &out, nullptr);

    // just for case we free memory which could be allocated by triangulate but we do not need (some of this can be
    // nullptr):
    trifree(out.pointattributelist);
    trifree(out.pointmarkerlist);
    trifree(out.triangleattributelist);
    trifree(out.trianglearealist);
    trifree(out.neighborlist);
    trifree(out.segmentlist);
    trifree(out.segmentmarkerlist);
    trifree(out.holelist);
    trifree(out.regionlist);
    trifree(out.edgelist);
    trifree(out.edgemarkerlist);
    trifree(out.normlist);

    // this will free rest of memory allocated by triangulate (even if an exception will be thrown):
    std::unique_ptr<REAL[], TrifreeCaller> out_points(out.pointlist);
    std::unique_ptr<int[], TrifreeCaller> out_triangles(out.trianglelist);

    shared_ptr<TriangularMesh2D> result = make_shared<TriangularMesh2D>();
    result->nodes.reserve(out.numberofpoints);
    for (std::size_t i = 0; i < std::size_t(out.numberofpoints) * 2; i += 2)
        result->nodes.emplace_back(out.pointlist[i], out.pointlist[i + 1]);
    result->elementNodes.reserve(out.numberoftriangles);
    for (std::size_t i = 0; i < std::size_t(out.numberoftriangles) * 3; i += 3)
        result->elementNodes.push_back({std::size_t(out.trianglelist[i]), std::size_t(out.trianglelist[i + 1]),
                                        std::size_t(out.trianglelist[i + 2])});
    // it is also possible by triangle, to use 6 numbers per triangle, but we do not support such things at the moment
    return result;
}

std::string TriangleGenerator::getSwitches() const {
    std::ostringstream result;

    // p - specifies vertices, segments, holes, regional attributes, and regional area constraints
    // z - points (and other items) are numbered from zero
    // Q - quiet
    // B - suppresses boundary markers in the output
    // P - suppresses the output .poly file
    result << "pzQqBP";

    // D - Conforming Delaunay triangulation
    if (delaunay) result << 'D';

    // a - imposes a maximum triangle area
    if (maxTriangleArea) result << 'a' << std::fixed << *maxTriangleArea;

    // q -  adds vertices to the mesh to ensure that all angles are between given and 140 degrees.
    if (minTriangleAngle) {
        result << 'q';
        if (!isnan(*minTriangleAngle)) result << std::fixed << *minTriangleAngle;
    }

    // TODO more configuration

    return result.str();
}

shared_ptr<MeshGenerator> readTriangleGenerator(XMLReader& reader, const Manager&) {
    shared_ptr<TriangleGenerator> result = make_shared<TriangleGenerator>();
    if (reader.requireTagOrEnd("options")) {
        result->maxTriangleArea = reader.getAttribute<double>("maxarea");
        result->minTriangleAngle = reader.getAttribute<double>("minangle");
        result->delaunay = reader.getAttribute<bool>("delaunay", false);
        result->full = reader.getAttribute<bool>("full", false);
        reader.requireTagEnd();  // end of options
        reader.requireTagEnd();  // end of generator
    }
    return result;
}

static RegisterMeshGeneratorReader trianglegenerator_reader("triangular2d.triangle", readTriangleGenerator);

}  // namespace plask
