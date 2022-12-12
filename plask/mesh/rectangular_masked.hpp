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
