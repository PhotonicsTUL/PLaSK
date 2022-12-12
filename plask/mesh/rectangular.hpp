/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
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
