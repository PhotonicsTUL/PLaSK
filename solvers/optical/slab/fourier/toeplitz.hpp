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
#ifndef PLASK__SOLVER_SLAB_TOEPLITZ_H
#define PLASK__SOLVER_SLAB_TOEPLITZ_H

#include <plask/plask.hpp>
#include "../matrices.hpp"

namespace plask { namespace optical { namespace slab {

/**
 * Solve T X = Y equation.
 *
 * \param TM Toeplitz matrix data stored as: first column + first row reversed (T00 T10 T20 T02 T01)
 * \param[inout] X on input right-hand size (Y), on output solution (X)
 */
template <typename D>
void ToeplitzLevinson(const DataVector<D>& TM, Matrix<D>& X)
{
    assert(TM.size() % 2 == 1);
    const size_t n = (TM.size() + 1) / 2;
    const size_t n2 = TM.size() - 1;
    const size_t nx = X.cols();

    std::unique_ptr<D[]> F(new D[n]);
    std::unique_ptr<D[]> B(new D[n]);
    std::unique_ptr<D[]> eX(new D[nx]);

    if (TM[0] == 0.) throw ComputationError("ToeplitzLevinson", "cannot invert Fourier coefficients matrix");

    F[0] = 1. / TM[0];
    B[0] = 1. / TM[0];
    for (size_t r = 0, dx = 0; r < nx; ++r, dx += n)
        X[dx] /= TM[0];

    for (size_t i = 1; i < n; i++) {
        F[i] = B[i] = 0.;
        D ef = 0., eb = 0.;

        for (size_t r = 0; r < nx; r++) eX[r] = 0.;

        for (size_t j = 0; j < i; j++) {
            size_t ij = i-j;
            ef += TM[ij] * F[j];
            eb += TM[n2-j] * B[j];
            for (size_t r = 0, mj = j; r < nx; ++r, mj += n)
                eX[r] += TM[ij] * X[mj];
        }

        D scal = (1. - ef * eb);
        if (scal == 0.) throw ComputationError("ToeplitzLevinson", "cannot invert Fourier coefficients matrix");
        scal = 1. / scal;

        D b = B[0];
        B[0] = -(eb * scal) * F[0];
        F[0] *= scal;
        for (size_t j = 1; j <= i; j++) {
            D f = F[j];
            D bj = B[j];
            F[j] = (scal * f) - ((ef * scal) * b);
            B[j] = (scal * b) - ((eb * scal) * f);
            b = bj;
        }

        for (size_t r = 0, dx = 0; r < nx; ++r, dx += n) {
            size_t ix = dx + i;
            for (size_t j = 0; j < i; j++)
                X[dx+j] += (X[ix] - eX[r]) * B[j];
            X[ix] = (X[ix] - eX[r]) * B[i];
        }
    }
}



}}} // namespace plask::optical::slab
#endif // PLASK__SOLVER_SLAB_TOEPLITZ_H
