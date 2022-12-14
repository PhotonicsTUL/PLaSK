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
#include "patterson.hpp"
#include "patterson-data.hpp"

namespace plask { namespace optical { namespace slab {

template <typename S, typename T>
S patterson(const std::function<S(T)>& fun, T a, T b, double& err, unsigned* order)
{
    double eps = err;
    err *= 2.;

    S result, result2;
    T D = (b - a) / 2., Z = (a + b) / 2.;

    S values[256]; std::fill_n(values, 256, 0.);
    values[0] = fun(Z);
    result = (b - a) * values[0];

    unsigned n;

    for (n = 1; err > eps && n < 9; ++n) {
        const unsigned N = 1 << n; // number of points in current iteration on one side of zero
        const unsigned stp = 256 >> n; // step in x array
        result2 = result;
        // Compute integral on scaled range [-1, +1]
        result = patterson_weights[n][0] * values[255];
        for (unsigned i = stp, j = 1; j < N; i += stp, ++j) {
            if (j % 2) { // only odd points are new ones
                const double x = patterson_points[i];
                T z1 = Z - D * x;
                T z2 = Z + D * x;
                values[i] = fun(z1) + fun(z2);
            }
            result += patterson_weights[n][j] * values[i];
        }
        // Rescale integral and compute error
        result *= D;
        err = abs(1. - result2 / result);
    }

#ifndef NDEBUG
    writelog(LOG_DEBUG, "Patterson quadrature for {0} points, error = {1}", (1<<n)-1, err);
#endif

    if (order) *order = n - 1;
    
    return result;
}

template double patterson<double,double>(const std::function<double(double)>& fun, double a, double b, double& err, unsigned* order);

}}} // namespace plask::optical::effective
