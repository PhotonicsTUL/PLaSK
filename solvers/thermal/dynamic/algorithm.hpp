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
#ifndef ALGORITHM_H
#define ALGORITHM_H


namespace plask { namespace thermal { namespace dynamic {

/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_CHOLESKY, ///< Cholesky factorization
    ALGORITHM_GAUSS    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
//    ALGORITHM_ITERATIVE ///< Conjugate gradient iterative solver
};

}}}


#endif // ALGORITHM_H
