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
#include "gauss_legendre.hpp"
#include "fortran.hpp"

#include <boost/math/special_functions/legendre.hpp>
using boost::math::legendre_p;

namespace plask { namespace optical { namespace slab {


void gaussLegendre(size_t n, std::vector<double>& abscissae, DataVector<double>& weights)
{
    int info;

    abscissae.assign(n, 0.);
    weights.reset(n);

    for (size_t i = 1; i != n; ++i)
        weights[i-1] = 0.5 / std::sqrt(1. - 0.25/double(i*i));

    dsterf(int(n), &abscissae.front(), weights.data(), info);
    if (info < 0) throw CriticalException("Gauss-Legendre abscissae: argument {:d} of DSTERF has bad value", -info);
    if (info > 0) throw ComputationError("Gauss-Legendre abscissae", "could not converge in {:d}-th element", info);

    double nn = double(n*n);
    auto x = abscissae.begin();
    auto w = weights.begin();
    for (; x != abscissae.end(); ++x, ++w) {
        double P = legendre_p(int(n-1), *x);
        *w = 2. * (1. - (*x)*(*x)) / (nn * P*P);
    }
}

}}} // # namespace plask::optical::slab
