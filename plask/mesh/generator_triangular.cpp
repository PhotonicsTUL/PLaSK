#include "generator_triangular.h"
#include "triangular2d.h"

extern "C" {
#include <triangle.h>
}

namespace plask {

struct TrifreeCaller {
    void operator()(void* ptr) const { trifree(ptr); }
};

shared_ptr<MeshD<2>> TriangleGenerator::generate(const shared_ptr<GeometryObjectD<DIM> > &geometry) {
    triangulateio in = {}, out = {};    // are fields are nulled, so we will only fill fields we need

    in.numberofpoints = 4;
    std::unique_ptr<REAL[]> in_points(new REAL[8]);
    Box2D bb = geometry->getBoundingBox();
    in.pointlist = in_points.get();
    in_points[0] = in_points[6] = bb.left();
    in_points[2] = in_points[4] = bb.right();
    in_points[1] = in_points[3] = bb.top();
    in_points[5] = in_points[7] = bb.bottom();
    in.numberofsegments = 4;
    std::unique_ptr<int[]> in_segments(new int[8]);
    in.segmentlist = in_segments.get();
    in_segments[0] = 0; in_segments[1] = 1;
    in_segments[2] = 1; in_segments[3] = 2;
    in_segments[4] = 2; in_segments[5] = 3;
    in_segments[6] = 3; in_segments[7] = 0;

    triangulate(const_cast<char*>(getSwitches().c_str()), &in, &out, nullptr);

    // just for case we free memory which could be allocated by triangulate but we do not need (some of this can be nullptr):
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

    // this will free rest of memory allocated by triangulate (even if an exception will be throwed):
    std::unique_ptr<REAL[], TrifreeCaller> out_points(out.pointlist);
    std::unique_ptr<int[], TrifreeCaller> out_triangles(out.trianglelist);

    shared_ptr<TriangularMesh2D> result = make_shared<TriangularMesh2D>();
    result->nodes.reserve(out.numberofpoints);
    for (std::size_t i = 0; i < std::size_t(out.numberofpoints)*2; i += 2)
        result->nodes.emplace_back(out.pointlist[i], out.pointlist[i+1]);
    result->elementNodes.reserve(out.numberoftriangles);
    for (std::size_t i = 0; i < std::size_t(out.numberoftriangles)*3; i += 3)
        result->elementNodes.push_back({
            std::size_t(out.trianglelist[i]),
            std::size_t(out.trianglelist[i+1]),
            std::size_t(out.trianglelist[i+2])
        });
        // it is also possible by triangle, to use 6 numbers per triangle, but we do not support such things at the moment
    return result;
}

std::string TriangleGenerator::getSwitches() const {
    std::ostringstream result;

    // z - points (and other items) are numbered from zero
    // Q - quiet
    // B - suppresses boundary markers in the output
    // P - suppresses the output .poly file. Saves disk space, but you lose the ability to maintain
    //     constraining segments on later refinements of the mesh.
    result << "zQBP";

    if (maxTriangleArea) result << 'a' << std::fixed << *maxTriangleArea;
    if (minTriangleAngle) {
        result << 'q';
        if (!isnan(*minTriangleAngle)) result << std::fixed << *minTriangleAngle;
    }

    // TODO more configuration

    return result.str();
}

shared_ptr<MeshGenerator> readTriangleGenerator(XMLReader& reader, const Manager&) {
    shared_ptr<TriangleGenerator> result = make_shared<TriangleGenerator>();
    result->maxTriangleArea = reader.getAttribute<double>("maxarea");
    result->minTriangleAngle = reader.getAttribute<double>("minangle");
    reader.requireTagEnd();
    return result;
}

static RegisterMeshGeneratorReader trianglegenerator_reader("triangle", readTriangleGenerator);

}   // namespace plask
