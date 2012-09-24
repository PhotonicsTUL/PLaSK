#ifndef PLASK__VECTOR__COMMON_H
#define PLASK__VECTOR__COMMON_H

/** @file
In this file some basis common for all vectors (2D and 3D) are defined.
*/

namespace plask {

/// Generic template for 2D and 3D vectors
template <int dim, typename T=double>
struct Vec {};

namespace axis {
    const std::size_t lon_index = 0;
    const std::size_t tran_index = 1;
    const std::size_t up_index = 2;
}   // axis

}   // namespace plask

#endif // PLASK__VECTOR__COMMON_H
