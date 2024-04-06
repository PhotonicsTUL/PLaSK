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
#include "muller.hpp"
#include "rootdigger-impl.hpp"
#include "solver.hpp"
using namespace std;

namespace plask { namespace optical { namespace slab {

//**************************************************************************
/// Search for a single mode starting from the given point: point
dcomplex RootMuller::find(dcomplex start)
{
    dcomplex first = start - 0.5 * params.initial_dist;
    dcomplex second = start + 0.5 * params.initial_dist;

    writelog(LOG_DETAIL, "Searching for the root with Muller method between {0} and {1}", str(first), str(second));
    log_value.resetCounter();

    double xtol2 = params.tolx * params.tolx;
    double fmin2 = params.tolf_min * params.tolf_min;
    double fmax2 = params.tolf_max * params.tolf_max;

    dcomplex x2 = first, x1 = second, x0 = start;

    dcomplex f2 = valFunction(x2); log_value(x2, f2);
    dcomplex f1 = valFunction(x1); log_value(x1, f1);
    dcomplex f0 = valFunction(x0); log_value.count(x0, f0);

    for (int i = 0; i < params.maxiter; ++i) {
        if (isnan(real(f0)) || isnan(imag(f0)))
            throw ComputationError(solver.getId(), "computed value is NaN");
        dcomplex q = (x0 - x1) / (x1 - x2);
        dcomplex A = q * f0 - q*(q+1.) * f1 + q*q * f2;
        dcomplex B = (2.*q+1.) * f0 - (q+1.)*(q+1.) * f1 + q*q * f2;
        dcomplex C = (q+1.) * f0;
        dcomplex S = sqrt(B*B - 4.*A*C);
        x2 = x1; f2 = f1;
        x1 = x0; f1 = f0;
        x0 = x1 - (x1-x2) * ( 2.*C / std::max(B+S, B-S, [](const dcomplex& a, const dcomplex& b){return abs2(a) < abs2(b);}) );
        f0 = valFunction(x0); log_value.count(x0, f0);
        if (abs2(f0) < fmin2 || (abs2(x0-x1) < xtol2 && abs2(f0) < fmax2)) {
            writelog(LOG_RESULT, "Found root at " + str(x0));
            return x0;
        }
    }

    throw ComputationError(solver.getId(), "Muller: {0}: maximum number of iterations reached", log_value.chartName());
}

}}} // namespace plask::optical::slab
