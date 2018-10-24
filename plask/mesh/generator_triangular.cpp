#include "generator_triangular.h"

extern "C" {
#include <triangle.h>
}

namespace plask {

shared_ptr<MeshD<2>> TriangleGenerator::generate(const shared_ptr<GeometryObjectD<DIM> > &geometry) {
    triangulateio in, out;
    triangulate(const_cast<char*>(getSwitches().c_str()), &in, &out, nullptr);
}

std::string TriangleGenerator::getSwitches() const {
    // z - points (and other items) are numbered from zero
    // Q - quiet
    std::string result = "zQ";
    // TODO configuration
    return result;
}

}   // namespace plask
