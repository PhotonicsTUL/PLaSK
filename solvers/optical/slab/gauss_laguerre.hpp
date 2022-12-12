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
#ifndef PLASK__SOLVER__SLAB_GAUSS_LAGUERRE_H
#define PLASK__SOLVER__SLAB_GAUSS_LAGUERRE_H

#include <plask/plask.hpp>

namespace plask { namespace optical { namespace slab {

/**
 * Compute ascissae and weights for Gauss-Laguerre quadatures.
 * \param n quadrature order
 * \param[out] abscissae computed abscissae
 * \param[out] weights corresponding weights
 * \param[in] scale scale parameter in the $\exp(sx)$ weight
 */
void gaussLaguerre(size_t n, std::vector<double>& abscissae, DataVector<double>& weights, double scale=1.);

}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_GAUSS_LAGUERRE_H
