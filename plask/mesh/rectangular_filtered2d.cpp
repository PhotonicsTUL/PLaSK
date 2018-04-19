#include "rectangular_filtered2d.h"

namespace plask {

bool RectangularFilteredMesh2D::prepareInterpolation(const Vec<2> &point, Vec<2> &wrapped_point, std::size_t &index0_lo, std::size_t &index0_hi, std::size_t &index1_lo, std::size_t &index1_hi, std::size_t &rectmesh_index_lo, const InterpolationFlags &flags) const {
    wrapped_point = flags.wrap(point);

    if (!canBeIncluded(wrapped_point)) return false;

    findIndexes(*rectangularMesh.axis0, wrapped_point.c0, index0_lo, index0_hi);
    findIndexes(*rectangularMesh.axis1, wrapped_point.c1, index1_lo, index1_hi);

    rectmesh_index_lo = rectangularMesh.index(index0_lo, index1_lo);
    return elementsSet.includes(rectangularMesh.getElementIndexFromLowIndex(rectmesh_index_lo));
}

}   // namespace plask
