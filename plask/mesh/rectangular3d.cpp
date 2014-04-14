#include "rectangular3d.h"

#include "regular1d.h"
#include "rectilinear1d.h"

namespace plask {

#define RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(first, second, third) \
    static std::size_t index_##first##second##third(const RectangularMesh<3>* mesh, std::size_t index0, std::size_t index1, std::size_t index2) { \
        return index##third + mesh->axis##third->size() * (index##second + mesh->axis##second->size() * index##first); \
    } \
    static std::size_t index##first##_##first##second##third(const RectangularMesh<3>* mesh, std::size_t mesh_index) { \
        return mesh_index / mesh->axis##third->size() / mesh->axis##second->size(); \
    } \
    static std::size_t index##second##_##first##second##third(const RectangularMesh<3>* mesh, std::size_t mesh_index) { \
        return (mesh_index / mesh->axis##third->size()) % mesh->axis##second->size(); \
    } \
    static std::size_t index##third##_##first##second##third(const RectangularMesh<3>* mesh, std::size_t mesh_index) { \
        return mesh_index % mesh->axis##third->size(); \
    }

RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(0,1,2)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(0,2,1)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(1,0,2)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(1,2,0)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(2,0,1)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(2,1,0)


void RectangularMesh<3>::setIterationOrder(IterationOrder iterationOrder) {
#   define RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(o1,o2,o3) \
        case ORDER_##o1##o2##o3: \
            index_f = index_##o1##o2##o3; index0_f = index0_##o1##o2##o3; \
            index1_f = index1_##o1##o2##o3; index2_f = index2_##o1##o2##o3; \
            major_axis = axis##o1.get(); medium_axis = axis##o2.get(); minor_axis = axis##o3.get(); \
            break;
    switch (iterationOrder) {
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(0,1,2)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(0,2,1)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(1,0,2)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(1,2,0)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(2,0,1)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(2,1,0)
        default:
            index_f = index_210; index0_f = index0_210;  index1_f = index1_210; index2_f = index2_210;
            major_axis = axis2.get(); medium_axis = axis1.get(); minor_axis = axis0.get();
    }
    this->fireChanged();
}


typename RectangularMesh<3>::IterationOrder RectangularMesh<3>::getIterationOrder() const {
    return this->index_f == decltype(this->index_f)(index_012) ? ORDER_012 :
           this->index_f == decltype(this->index_f)(index_021) ? ORDER_021 :
           this->index_f == decltype(this->index_f)(index_102) ? ORDER_102 :
           this->index_f == decltype(this->index_f)(index_120) ? ORDER_120 :
           this->index_f == decltype(this->index_f)(index_201) ? ORDER_201 :
                                                                 ORDER_210;
}

void RectangularMesh<3>::setOptimalIterationOrder() {
#   define RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(first, second, third) \
        if (this->axis##third->size() <= this->axis##second->size() && this->axis##second->size() <= this->axis##first->size()) { \
            setIterationOrder(ORDER_##first##second##third); return; \
        }
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(0,1,2)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(0,2,1)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(1,0,2)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(1,2,0)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(2,0,1)
    RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(2,1,0)
}


// Particular instantations
template class RectangularMesh<3>;


} // namespace plask
