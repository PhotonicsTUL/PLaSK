#include "rectangular3d.h"

#include "regular1d.h"
#include "rectilinear1d.h"

namespace plask {

#define RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(first, second, third) \
    template <typename Mesh1D> \
    static std::size_t index_##first##second##third(const RectangularMesh<3,Mesh1D>* mesh, std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) { \
        return c##third##_index + mesh->c##third.size() * (c##second##_index + mesh->c##second.size() * c##first##_index); \
    } \
    template <typename Mesh1D> \
    static std::size_t index##first##_##first##second##third(const RectangularMesh<3,Mesh1D>* mesh, std::size_t mesh_index) { \
        return mesh_index / mesh->c##third.size() / mesh->c##second.size(); \
    } \
    template <typename Mesh1D> \
    static std::size_t index##second##_##first##second##third(const RectangularMesh<3,Mesh1D>* mesh, std::size_t mesh_index) { \
        return (mesh_index / mesh->c##third.size()) % mesh->c##second.size(); \
    } \
    template <typename Mesh1D> \
    static std::size_t index##third##_##first##second##third(const RectangularMesh<3,Mesh1D>* mesh, std::size_t mesh_index) { \
        return mesh_index % mesh->c##third.size(); \
    }

RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(0,1,2)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(0,2,1)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(1,0,2)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(1,2,0)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(2,0,1)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(2,1,0)


template <typename Mesh1D>
void RectangularMesh<3,Mesh1D>::setIterationOrder(IterationOrder iterationOrder) {
#   define RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(o1,o2,o3) \
        case ORDER_##o1##o2##o3: \
            index_f = index_##o1##o2##o3; index0_f = index0_##o1##o2##o3; \
            index1_f = index1_##o1##o2##o3; index2_f = index2_##o1##o2##o3; \
            major_axis = &c##o1; middle_axis = &c##o2; minor_axis = &c##o3; \
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
            major_axis = &c2; middle_axis = &c1; minor_axis = &c0;
    }
    this->fireChanged();
}


template <typename Mesh1D>
typename RectangularMesh<3,Mesh1D>::IterationOrder RectangularMesh<3,Mesh1D>::getIterationOrder() const {
    return this->index_f == decltype(this->index_f)(index_012) ? ORDER_012 :
           this->index_f == decltype(this->index_f)(index_021) ? ORDER_021 :
           this->index_f == decltype(this->index_f)(index_102) ? ORDER_102 :
           this->index_f == decltype(this->index_f)(index_120) ? ORDER_120 :
           this->index_f == decltype(this->index_f)(index_201) ? ORDER_201 :
                                                                 ORDER_210;
}

template <typename Mesh1D>
void RectangularMesh<3,Mesh1D>::setOptimalIterationOrder() {
#   define RECTANGULAR_MESH_3D_DETERMINE_ITERATION_ORDER(first, second, third) \
        if (this->c##third.size() <= this->c##second.size() && this->c##second.size() <= this->c##first.size()) { \
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
template class RectangularMesh<3,RegularMesh1D>;
template class RectangularMesh<3,RectilinearMesh1D>;


} // namespace plask
