#include "rectilinear2d.h"

namespace plask {

static std::size_t normal_index(const RectilinearMesh2d* mesh, std::size_t c0_index, std::size_t c1_index) {
    return c0_index + mesh->c0.size() * c1_index;
}
static std::size_t normal_index0(const RectilinearMesh2d* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->c0.size();
}
static std::size_t normal_index1(const RectilinearMesh2d* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->c0.size();
}

static std::size_t transposed_index(const RectilinearMesh2d* mesh, std::size_t c0_index, std::size_t c1_index) {
    return mesh->c1.size() * c0_index + c1_index;
}
static std::size_t transposed_index0(const RectilinearMesh2d* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->c1.size();
}
static std::size_t transposed_index1(const RectilinearMesh2d* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->c1.size();
}


void RectilinearMesh2d::setIterationOrder(IterationOrder iterationOrder) {
    if (iterationOrder == TRANSPOSED_ORDER) {
        index_f = transposed_index;
        index0_f = transposed_index0;
        index1_f = transposed_index1;
    } else {
        index_f = normal_index;
        index0_f = normal_index0;
        index1_f = normal_index1;
    }
}

RectilinearMesh2d::IterationOrder RectilinearMesh2d::getIterationOrder() const {
    return index_f == transposed_index ? TRANSPOSED_ORDER : NORMAL_ORDER;
}

void RectilinearMesh2d::setOptimalIterationOrder() {
    setIterationOrder(c0.size() > c1.size() ? TRANSPOSED_ORDER : NORMAL_ORDER);
}

void RectilinearMesh2d::buildFromGeometry(const GeometryElementD<2>& geometry) {
    std::vector<Box2d> boxes = geometry.getLeafsBoundingBoxes();

    for (auto box: boxes) {
        c0.addPoint(box.lower.c0);
        c0.addPoint(box.upper.c0);
        c1.addPoint(box.lower.c1);
        c1.addPoint(box.upper.c1);
    }
}

RectilinearMesh2d RectilinearMesh2d::getMidpointsMesh() const {

    if (c0.size() < 2 || c1.size() < 2) throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1d line0;
    for (auto a = c0.begin(), b = c0.begin()+1; b != c0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearMesh1d line1;
    for (auto a = c1.begin(), b = c1.begin()+1; b != c1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    return RectilinearMesh2d(line0, line1, getIterationOrder());
}


} // namespace plask
