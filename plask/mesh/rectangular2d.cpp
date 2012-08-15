#include "rectangular2d.h"

#include "regular1d.h"
#include "rectilinear1d.h"

namespace plask {

template <typename Mesh1D>
static std::size_t normal_index(const RectangularMesh<2,Mesh1D>* mesh, std::size_t c0_index, std::size_t c1_index) {
    return c0_index + mesh->c0.size() * c1_index;
}
template <typename Mesh1D>
static std::size_t normal_index0(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->c0.size();
}
template <typename Mesh1D>
static std::size_t normal_index1(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->c0.size();
}

template <typename Mesh1D>
static std::size_t transposed_index(const RectangularMesh<2,Mesh1D>* mesh, std::size_t c0_index, std::size_t c1_index) {
    return mesh->c1.size() * c0_index + c1_index;
}
template <typename Mesh1D>
static std::size_t transposed_index0(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->c1.size();
}
template <typename Mesh1D>
static std::size_t transposed_index1(const RectangularMesh<2,Mesh1D>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->c1.size();
}

template <typename Mesh1D>
void RectangularMesh<2,Mesh1D>::setIterationOrder(IterationOrder iterationOrder) {
    if (iterationOrder == TRANSPOSED_ORDER) {
        index_f = transposed_index<Mesh1D>;
        index0_f = transposed_index0<Mesh1D>;
        index1_f = transposed_index1<Mesh1D>;
    } else {
        index_f = normal_index<Mesh1D>;
        index0_f = normal_index0<Mesh1D>;
        index1_f = normal_index1<Mesh1D>;
    }
    fireChanged();
}

template <typename Mesh1D>
typename RectangularMesh<2,Mesh1D>::IterationOrder RectangularMesh<2,Mesh1D>::getIterationOrder() const {
    return (index_f == &transposed_index<Mesh1D>)? TRANSPOSED_ORDER : NORMAL_ORDER;
}

// Particular instantations
template class RectangularMesh<2,RectilinearMesh1D>;
template class RectangularMesh<2,RegularMesh1D>;

} // namespace plask
