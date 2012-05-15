#include "rectilinear3d.h"

namespace plask {

#define DECLARE_ITERATION_ORDER(first, second, third) \
    static std::size_t index_##first##second##third(const RectilinearMesh3d* mesh, std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) { \
        return c##first##_index + mesh->c##first.size() * (c##second##_index + mesh->c##second.size() * c##third##_index); \
    } \
    static std::size_t index##first##_##first##second##third(const RectilinearMesh3d* mesh, std::size_t mesh_index) { \
        return mesh_index % mesh->c##first.size(); \
    } \
    static std::size_t index##second##_##first##second##third(const RectilinearMesh3d* mesh, std::size_t mesh_index) { \
        return (mesh_index / mesh->c##first.size()) % mesh->c##second.size(); \
    } \
    static std::size_t index##third##_##first##second##third(const RectilinearMesh3d* mesh, std::size_t mesh_index) { \
        return mesh_index / mesh->c##first.size() / mesh->c##second.size(); \
    }

DECLARE_ITERATION_ORDER(0,1,2)
DECLARE_ITERATION_ORDER(0,2,1)
DECLARE_ITERATION_ORDER(1,0,2)
DECLARE_ITERATION_ORDER(1,2,0)
DECLARE_ITERATION_ORDER(2,0,1)
DECLARE_ITERATION_ORDER(2,1,0)


void RectilinearMesh3d::setIterationOrder(IterationOrder iterationOrder) {
#   define CASE_ITERATION_ORDER(order) \
        case ORDER_##order: index_f = index_##order; index0_f = index0_##order;  index1_f = index1_##order; index2_f = index2_##order; return;
    switch (iterationOrder) {
        CASE_ITERATION_ORDER(021)
        CASE_ITERATION_ORDER(102)
        CASE_ITERATION_ORDER(120)
        CASE_ITERATION_ORDER(201)
        CASE_ITERATION_ORDER(210)
        default:
            index_f = index_012; index0_f = index0_012;  index1_f = index1_012; index2_f = index2_012; return;
    }
}


RectilinearMesh3d::IterationOrder RectilinearMesh3d::getIterationOrder() const {
    return index_f == index_012 ? ORDER_012 :
           index_f == index_021 ? ORDER_021 :
           index_f == index_102 ? ORDER_102 :
           index_f == index_120 ? ORDER_120 :
           index_f == index_201 ? ORDER_201 :
                                  ORDER_210;
}


void RectilinearMesh3d::setOptimalIterationOrder() {
#   define DETERMINE_ITERATION_ORDER(first, second, third) \
        if (c##first.size() <= c##second.size() && c##second.size() <= c##third.size()) { setIterationOrder(ORDER_##first##second##third); return; }
    DETERMINE_ITERATION_ORDER(0,1,2)
    DETERMINE_ITERATION_ORDER(0,2,1)
    DETERMINE_ITERATION_ORDER(1,0,2)
    DETERMINE_ITERATION_ORDER(1,2,0)
    DETERMINE_ITERATION_ORDER(2,0,1)
    DETERMINE_ITERATION_ORDER(2,1,0)
}


void RectilinearMesh3d::buildFromGeometry(const GeometryElementD<3>& geometry) {
    std::vector<Box3d> boxes = geometry.getLeafsBoundingBoxes();

    for (auto box: boxes) {
        c0.addPoint(box.lower.c0);
        c0.addPoint(box.upper.c0);
        c1.addPoint(box.lower.c1);
        c1.addPoint(box.upper.c1);
        c2.addPoint(box.lower.c2);
        c2.addPoint(box.upper.c2);
    }
}


RectilinearMesh3d RectilinearMesh3d::getMidpointsMesh() const {

    if (c0.size() < 2 || c1.size() < 2 || c2.size() < 2) throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1d line0;
    for (auto a = c0.begin(), b = c0.begin()+1; b != c0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearMesh1d line1;
    for (auto a = c1.begin(), b = c1.begin()+1; b != c1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    RectilinearMesh1d line2;
    for (auto a = c2.begin(), b = c2.begin()+1; b != c2.end(); ++a, ++b)
        line2.addPoint(0.5 * (*a + *b));

    return RectilinearMesh3d(line0, line1, line2, getIterationOrder());
}



} // namespace plask
