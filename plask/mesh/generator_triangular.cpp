#include "generator_triangular.h"
#include "triangular2d.h"

extern "C" {
#include <triangle.h>
}

namespace plask {

shared_ptr<MeshD<2>> TriangleGenerator::generate(const shared_ptr<GeometryObjectD<DIM> > &geometry) {
    triangulateio in, out;
    // TODO prepare in and out
    triangulate(const_cast<char*>(getSwitches().c_str()), &in, &out, nullptr);

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

}   // namespace plask
