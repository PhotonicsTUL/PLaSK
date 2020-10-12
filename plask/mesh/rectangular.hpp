#ifndef PLASK__RECTANGULAR_H
#define PLASK__RECTANGULAR_H

#include "axis1d.hpp"
#include "rectangular2d.hpp"
#include "rectangular3d.hpp"

#include "ordered1d.hpp"
#include "regular1d.hpp"

namespace plask {

template <int DIM>
using RectangularMesh =
    typename std::conditional<
        DIM == 2,
        RectangularMesh2D,
        typename std::conditional<DIM == 3, RectangularMesh3D, void>::type
    >::type;

template <int dim>
struct Rectangular_t {
    typedef RectangularMesh<dim> Rectangular;
    typedef RectangularMesh<dim> Regular;
    typedef RectangularMesh<dim> Rectilinear;
};

template <>
struct Rectangular_t<1> {
    typedef MeshAxis Rectangular;
    typedef RegularAxis Regular;
    typedef OrderedAxis Rectilinear;
};

}   // namespace plask



#endif // PLASK__RECTANGULAR_H
