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
#include "expansion.hpp"

namespace plask { namespace optical { namespace slab {

void Expansion::getDiagonalEigenvectors(cmatrix& Te, cmatrix& Te1, const cmatrix&, const cdiagonal&)
{
    size_t nr = Te.rows(), nc = Te.cols();
    // Eigenvector matrix is simply a unity matrix
    std::fill_n(Te.data(), nr*nc, 0.);
    std::fill_n(Te1.data(), nr*nc, 0.);
    for (size_t i = 0; i < nc; i++)
        Te(i,i) = Te1(i,i) = 1.;
}

}}} // namespace
