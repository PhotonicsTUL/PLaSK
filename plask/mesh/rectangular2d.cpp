#include "rectangular2d.h"

#include "regular1d.h"
#include "rectilinear1d.h"

namespace plask {

template <typename Mesh1D>
static std::size_t normal_index(const RectangularMesh<2,Mesh1D>* mesh, std::size_t index0, std::size_t index1) {
    return index0 + mesh->axis0.size() * index1;
}
template <typename Mesh1D>
static std::size_t normal_index0(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis0.size();
}
template <typename Mesh1D>
static std::size_t normal_index1(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis0.size();
}

template <typename Mesh1D>
static std::size_t transposed_index(const RectangularMesh<2,Mesh1D>* mesh, std::size_t index0, std::size_t index1) {
    return mesh->axis1.size() * index0 + index1;
}
template <typename Mesh1D>
static std::size_t transposed_index0(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis1.size();
}
template <typename Mesh1D>
static std::size_t transposed_index1(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis1.size();
}

template <typename Mesh1D>
void RectangularMesh<2,Mesh1D>::setIterationOrder(IterationOrder iterationOrder) {
    if (iterationOrder == TRANSPOSED_ORDER) {
        index_f = transposed_index<Mesh1D>;
        index0_f = transposed_index0<Mesh1D>;
        index1_f = transposed_index1<Mesh1D>;
        minor_axis = &axis1;
        major_axis = &axis0;
    } else {
        index_f = normal_index<Mesh1D>;
        index0_f = normal_index0<Mesh1D>;
        index1_f = normal_index1<Mesh1D>;
        minor_axis = &axis0;
        major_axis = &axis1;
    }
    this->fireChanged();
}

template <typename Mesh1D>
typename RectangularMesh<2,Mesh1D>::IterationOrder RectangularMesh<2,Mesh1D>::getIterationOrder() const {
    return (index_f == &transposed_index<Mesh1D>)? TRANSPOSED_ORDER : NORMAL_ORDER;
}

// Particular instantations
template class RectangularMesh<2,RectilinearMesh1D>;
template class RectangularMesh<2,RegularMesh1D>;

} // namespace plask
