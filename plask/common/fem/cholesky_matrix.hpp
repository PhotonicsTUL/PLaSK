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
#ifndef PLASK_COMMON_FEM_CHOLESKY_MATRIX_H
#define PLASK_COMMON_FEM_CHOLESKY_MATRIX_H

#include <cstddef>

#include "matrix.hpp"

#define UPLO 'L'

// BLAS routine to multiply matrix by vector
#define dsbmv F77_GLOBAL(dsbmv, DSBMV)
F77SUB dsbmv(const char& uplo,
             const int& n,
             const int& k,
             const double& alpha,
             const double* a,
             const int& lda,
             const double* x,
             const int& incx,
             const double& beta,
             double* y,
             const int& incy);  // y = alpha*A*x + beta*y,

// LAPACK routines to solve set of linear equations
#define dpbtrf F77_GLOBAL(dpbtrf, DPBTRF)
F77SUB dpbtrf(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);

#define dpbtrs F77_GLOBAL(dpbtrs, DPBTRS)
F77SUB dpbtrs(const char& uplo,
              const int& n,
              const int& kd,
              const int& nrhs,
              double* ab,
              const int& ldab,
              double* b,
              const int& ldb,
              int& info);

namespace plask {

/**
 * Symmetric band matrix structure.
 * Data is stored in LAPACK format.
 */
struct DpbMatrix : BandMatrix {
    /**
     * Create matrix
     * \param rank rank of the matrix
     * \param band maximum band size
     */
    DpbMatrix(const Solver* solver, size_t rank, size_t band)
        : BandMatrix(solver, rank, band, ((band + 1 + (15 / sizeof(double))) & ~size_t(15 / sizeof(double))) - 1) {}

    size_t index(size_t r, size_t c) override {
        assert(r < rank && c < rank);
        if (r < c) {
            assert(c - r <= kd);
//          if UPLO = 'U', AB(kd+i-j,j) = A(i,j) for max(0,j-kd)<=i<=j;
//          if UPLO = 'L', AB(i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
#if UPLO == 'U'
            return ld * c + r + kd;
#else
            return ld * r + c;
#endif
        } else {
            assert(r - c <= kd);
#if UPLO == 'U'
            return ld * r + c + kd;
#else
            return ld * c + r;
#endif
        }
    }

    void factorize() override {
        int info = 0;

        solver->writelog(LOG_DETAIL, "Factorizing system");

        dpbtrf(UPLO, int(rank), int(kd), data, int(ld + 1), info);
        if (info < 0)
            throw CriticalException("{0}: Argument {1} of `dpbtrf` has illegal value", solver->getId(), -info);
        else if (info > 0)
            throw ComputationError(solver->getId(), "Leading minor of order {0} of the stiffness matrix is not positive-definite",
                                   info);
    }

    void solverhs(DataVector<double>& B, DataVector<double>& X) override {
        solver->writelog(LOG_DETAIL, "Solving matrix system");

        int info = 0;
        dpbtrs(UPLO, int(rank), int(kd), 1, data, int(ld + 1), B.data(), int(B.size()), info);
        if (info < 0) throw CriticalException("{0}: Argument {1} of `dpbtrs` has illegal value", solver->getId(), -info);

        std::swap(B, X);
    }

    void mult(const DataVector<const double>& vector, DataVector<double>& result) override {
        dsbmv(UPLO, int(rank), int(kd), 1.0, data, int(ld) + 1, vector.data(), 1, 0.0, result.data(), 1);
    }

    void addmult(const DataVector<const double>& vector, DataVector<double>& result) override {
        dsbmv(UPLO, int(rank), int(kd), 1.0, data, int(ld) + 1, vector.data(), 1, 1.0, result.data(), 1);
    }
};

}  // namespace plask

#endif  // PLASK_COMMON_FEM_CHOLESKY_MATRIX_H
