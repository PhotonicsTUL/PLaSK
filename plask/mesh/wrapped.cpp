#include "wrapped.h"

namespace plask {

/**
 * Construct mirror adapter
 * \param original original mesh
 * \param sym_tran,sym_vert indicate whether there is any symmetry along given axis
 */
template <>
WrappedMesh<2>::WrappedMesh(const MeshD<2>& original, const shared_ptr<GeometryD<2>>& geometry): original(original) {
    symmetric[0] = geometry->isSymmetric(Geometry::DIRECTION_TRAN);
    symmetric[1] = geometry->isSymmetric(Geometry::DIRECTION_VERT);
}

/**
 * Construct mirror adapter
 * \param original original mesh
 * \param sym_long,sym_tran,sym_vert indicate whether there is any symmetry along given axis
 */
template <>
WrappedMesh<3>::WrappedMesh(const MeshD<3>& original, const shared_ptr<GeometryD<3>>& geometry): original(original) {
    symmetric[0] = geometry->isSymmetric(Geometry::DIRECTION_LONG);
    symmetric[1] = geometry->isSymmetric(Geometry::DIRECTION_TRAN);
    symmetric[2] = geometry->isSymmetric(Geometry::DIRECTION_VERT);
}

template struct WrappedMesh<2>;
template struct WrappedMesh<3>;


}