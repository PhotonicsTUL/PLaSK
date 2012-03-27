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

} // namespace plask
