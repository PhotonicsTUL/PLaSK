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
    for (std::size_t i = 0; i < std::size_t(out.numberofpoints)*2; i += 2)
        result->nodes.emplace_back(out.pointlist[i], out.pointlist[i+1]);
    for (std::size_t i = 0; i < std::size_t(out.numberoftriangles)*3; i += 3)
        result->elementsNodes.push_back({
            std::size_t(out.trianglelist[i]),
            std::size_t(out.trianglelist[i+1]),
            std::size_t(out.trianglelist[i+2])
        });
        // it is also possible by triangle, to use 6 numbers per triangle, but we do not support such things at the moment
    return result;
}

std::string TriangleGenerator::getSwitches() const {
    // z - points (and other items) are numbered from zero
    // Q - quiet
    std::string result = "zQ";
    // TODO configuration
    return result;
}

}   // namespace plask
