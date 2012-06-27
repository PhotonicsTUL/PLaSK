#include "rectangular3d.h"

namespace plask {

#define RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(first, second, third) \
    template <typename Mesh1D> \
    static std::size_t index_##first##second##third(const RectangularMesh3D<Mesh1D>* mesh, std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) { \
        return c##third##_index + mesh->c##third.size() * (c##second##_index + mesh->c##second.size() * c##first##_index); \
    } \
    template <typename Mesh1D> \
    static std::size_t index##first##_##first##second##third(const RectangularMesh3D<Mesh1D>* mesh, std::size_t mesh_index) { \
        return mesh_index / mesh->c##third.size() / mesh->c##second.size(); \
    } \
    template <typename Mesh1D> \
    static std::size_t index##second##_##first##second##third(const RectangularMesh3D<Mesh1D>* mesh, std::size_t mesh_index) { \
        return (mesh_index / mesh->c##third.size()) % mesh->c##second.size(); \
    } \
    template <typename Mesh1D> \
    static std::size_t index##third##_##first##second##third(const RectangularMesh3D<Mesh1D>* mesh, std::size_t mesh_index) { \
        return mesh_index % mesh->c##third.size(); \
    }

RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(0,1,2)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(0,2,1)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(1,0,2)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(1,2,0)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(2,0,1)
RECTANGULAR_MESH_3D_DECLARE_ITERATION_ORDER(2,1,0)


template <typename Mesh1D>
void RectangularMesh3D<Mesh1D>::setIterationOrder(IterationOrder iterationOrder) {
#   define RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(order) \
        case ORDER_##order: index_f = index_##order; index0_f = index0_##order;  index1_f = index1_##order; index2_f = index2_##order; break;
    switch (iterationOrder) {
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(021)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(102)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(120)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(201)
        RECTANGULAR_MESH_3D_CASE_ITERATION_ORDER(210)
        default:
            index_f = index_012; index0_f = index0_012;  index1_f = index1_012; index2_f = index2_012; break;
    }
    this->fireChanged();
}


template <typename Mesh1D>
typename RectangularMesh3D<Mesh1D>::IterationOrder RectangularMesh3D<Mesh1D>::getIterationOrder() const {
    return this->index_f == decltype(this->index_f)(index_012) ? ORDER_012 :
           this->index_f == decltype(this->index_f)(index_021) ? ORDER_021 :
           this->index_f == decltype(this->index_f)(index_102) ? ORDER_102 :
           this->index_f == decltype(this->index_f)(index_120) ? ORDER_120 :
           this->index_f == decltype(this->index_f)(index_201) ? ORDER_201 :
                                        ORDER_210;
}

template <typename Mesh1D>
void RectangularMesh3D<Mesh1D>::setOptimalIterationOrder() {
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





} // namespace plask
