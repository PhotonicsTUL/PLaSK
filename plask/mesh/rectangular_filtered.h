#ifndef PLASK__RECTANGULAR_FILTERED_H
#define PLASK__RECTANGULAR_FILTERED_H

#include <type_traits>

#include "rectangular_filtered2d.h"
#include "rectangular_filtered3d.h"

namespace plask {

template <int DIM>
using RectangularFilteredMesh =
    typename std::conditional<
        DIM == 2,
        RectangularFilteredMesh2D,
        typename std::conditional<DIM == 3, RectangularFilteredMesh3D, void>::type
    >::type;

}   // namespace plask

#endif // PLASK__RECTANGULAR_FILTERED_H
