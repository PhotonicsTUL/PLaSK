#ifndef PLASK__SPACE_H
#define PLASK__SPACE_H

#include "vector/2d.h"
#include "vector/3d.h"

namespace plask {

//TODO Vec -> typy przestrzeni udostępniające typ dla pkt.

/**
Provide type for vector in given space: double for 1d, etc.

Use example: <code>Cartesian<2>::type my_2dvec_obj;</code>
@tparam dim number of space dimentions
*/
template <int dim>
struct Cartesian {};

template <>
struct Cartesian<1> {
    typedef double PointType;
};

template <>
struct Cartesian<2> {
    typedef Vector2d<double> PointType;
};

template <>
struct Cartesian<3> {
    typedef Vector3d<double> PointType;
};

} // namespace plask

#endif  //PLASK__SPACE_H
