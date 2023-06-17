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
#ifndef PLASK__SOLVER__ELECTRICAL_SHOCKLEY_COMMON_H
#define PLASK__SOLVER__ELECTRICAL_SHOCKLEY_COMMON_H

#include <limits>

#include <plask/plask.hpp>
#include <plask/common/fem.hpp>


namespace plask { namespace electrical { namespace shockley {

/// Convergence algorithm
enum Convergence {
    CONVERGENCE_FAST = 0,   ///< Default fast convergence
    CONVERGENCE_STABLE = 1  ///< Stable slow convergence
};

}}} // # namespace plask::electrical::shockley

#endif // PLASK__SOLVER__ELECTRICAL_SHOCKLEY_COMMON_H
