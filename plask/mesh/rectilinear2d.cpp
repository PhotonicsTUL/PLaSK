#include "rectilinear2d.h"

namespace plask {

std::size_t normal_index(const RectilinearMesh2d* mesh, std::size_t c0_index, std::size_t c1_index) {
    return c0_index + mesh->c0.size() * c1_index;
}

std::size_t normal_index0(const RectilinearMesh2d* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->c0.size();
}

std::size_t normal_index1(const RectilinearMesh2d* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->c0.size();
}


std::size_t swapped_index(const RectilinearMesh2d* mesh, std::size_t c0_index, std::size_t c1_index) {
    return mesh->c1.size() * c0_index + c1_index;
}

std::size_t swapped_index0(const RectilinearMesh2d* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->c1.size();
}

std::size_t swapped_index1(const RectilinearMesh2d* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->c1.size();
}

void RectilinearMesh2d::setIterationOrder(IterationOrder iterationOrder) {
    if (iterationOrder == SWAPPED_ORDER) {
        index_f = swapped_index;
        index0_f = swapped_index0;
        index1_f = swapped_index1;
    } else {
        index_f = normal_index;
        index0_f = normal_index0;
        index1_f = normal_index1;
    }
}

RectilinearMesh2d::IterationOrder RectilinearMesh2d::getIterationOrder() const {
    return index_f == swapped_index ? SWAPPED_ORDER : NORMAL_ORDER;
}

void RectilinearMesh2d::setOptimumIterationOrder() {
    setIterationOrder(c0.size() > c1.size() ? SWAPPED_ORDER : NORMAL_ORDER);
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
