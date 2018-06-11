#include "rectangular_filtered3d.h"

namespace plask {

bool RectangularFilteredMesh3D::prepareInterpolation(const Vec<3> &point, Vec<3> &wrapped_point,
                                                     std::size_t& index0_lo, std::size_t& index0_hi,
                                                     std::size_t& index1_lo, std::size_t& index1_hi,
                                                     std::size_t& index2_lo, std::size_t& index2_hi,
                                                     std::size_t &rectmesh_index_lo, const InterpolationFlags &flags) const {
    wrapped_point = flags.wrap(point);

    if (!canBeIncluded(wrapped_point)) return false;

    findIndexes(*rectangularMesh.axis[0], wrapped_point.c0, index0_lo, index0_hi);
    findIndexes(*rectangularMesh.axis[1], wrapped_point.c1, index1_lo, index1_hi);
    findIndexes(*rectangularMesh.axis[2], wrapped_point.c2, index2_lo, index2_hi);

    rectmesh_index_lo = rectangularMesh.index(index0_lo, index1_lo, index2_lo);
    return elementsSet.includes(rectangularMesh.getElementIndexFromLowIndex(rectmesh_index_lo));
}

}   // namespace plask
