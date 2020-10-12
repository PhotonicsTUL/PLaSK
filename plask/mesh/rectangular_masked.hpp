#ifndef PLASK__RECTANGULAR_MASKED_H
#define PLASK__RECTANGULAR_MASKED_H

#include <type_traits>

#include "rectangular_masked2d.hpp"
#include "rectangular_masked3d.hpp"

namespace plask {

template <int DIM>
using RectangularMaskedMesh =
    typename std::conditional<
        DIM == 2,
        RectangularMaskedMesh2D,
        typename std::conditional<DIM == 3, RectangularMaskedMesh3D, void>::type
    >::type;

}   // namespace plask

#endif // PLASK__RECTANGULAR_MASKED_H
