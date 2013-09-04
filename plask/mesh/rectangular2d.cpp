#include "rectangular2d.h"

#include "regular1d.h"
#include "rectilinear1d.h"

namespace plask {

template <typename AxisT>
static std::size_t normal_index(const RectangularMesh<2,AxisT>* mesh, std::size_t index0, std::size_t index1) {
    return index0 + mesh->axis0.size() * index1;
}
template <typename AxisT>
static std::size_t normal_index0(const RectangularMesh<2,AxisT>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis0.size();
}
template <typename AxisT>
static std::size_t normal_index1(const RectangularMesh<2,AxisT>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis0.size();
}

template <typename AxisT>
static std::size_t transposed_index(const RectangularMesh<2,AxisT>* mesh, std::size_t index0, std::size_t index1) {
    return mesh->axis1.size() * index0 + index1;
}
template <typename AxisT>
static std::size_t transposed_index0(const RectangularMesh<2,AxisT>* mesh, std::size_t mesh_index) {
    return mesh_index / mesh->axis1.size();
}
template <typename AxisT>
static std::size_t transposed_index1(const RectangularMesh<2,AxisT>* mesh, std::size_t mesh_index) {
    return mesh_index % mesh->axis1.size();
}

template <typename AxisT>
void RectangularMesh<2,AxisT>::setIterationOrder(IterationOrder iterationOrder) {
    if (iterationOrder == ORDER_TRANSPOSED) {
        index_f = transposed_index<AxisT>;
        index0_f = transposed_index0<AxisT>;
        index1_f = transposed_index1<AxisT>;
        minor_axis = &axis1;
        major_axis = &axis0;
    } else {
        index_f = normal_index<AxisT>;
        index0_f = normal_index0<AxisT>;
        index1_f = normal_index1<AxisT>;
        minor_axis = &axis0;
        major_axis = &axis1;
    }
    this->fireChanged();
}

template <typename AxisT>
typename RectangularMesh<2,AxisT>::IterationOrder RectangularMesh<2,AxisT>::getIterationOrder() const {
    return (index_f == &transposed_index<AxisT>)? ORDER_TRANSPOSED : ORDER_NORMAL;
}

// Particular instantations
template class RectangularMesh<2,RectilinearAxis>;
template class RectangularMesh<2,RegularAxis>;

} // namespace plask
