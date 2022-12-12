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

#include <plask/plask.hpp>
#include <limits>

#include "block_matrix.hpp"
#include "gauss_matrix.hpp"
#include "conjugate_gradient.hpp"

namespace plask { namespace electrical { namespace shockley {

/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_CHOLESKY, ///< Cholesky factorization
    ALGORITHM_GAUSS,    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE ///< Conjugate gradient iterative solver
};

/// Convergence algorithm
enum Convergence {
    CONVERGENCE_FAST = 0,   ///< Default fast convergence
    CONVERGENCE_STABLE = 1  ///< Stable slown down convergence
};



}}} // # namespace plask::electrical::shockley

#endif // PLASK__SOLVER__ELECTRICAL_SHOCKLEY_COMMON_H
