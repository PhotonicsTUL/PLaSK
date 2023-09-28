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
#ifndef PLASK_COMMON_FEM_GAUSS_MATRIX_H
#define PLASK_COMMON_FEM_GAUSS_MATRIX_H

#include <cstddef>

#include "matrix.hpp"

// BLAS routine to multiply matrix by vector
#define dgbmv F77_GLOBAL(dgbmv, DGBMV)
F77SUB dgbmv(const char& trans,
             const int& m,
             const int& n,
             const int& kl,
             const int& ku,
             const double& alpha,
             double* a,
             const int& lda,
             const double* x,
             int incx,
             const double& beta,
             double* y,
             int incy);

// LAPACK routines to solve set of linear equations
#define dgbtrf F77_GLOBAL(dgbtrf, DGBTRF)
F77SUB dgbtrf(const int& m, const int& n, const int& kl, const int& ku, double* ab, const int& ldab, int* ipiv, int& info);

#define dgbtrs F77_GLOBAL(dgbtrs, DGBTRS)
F77SUB dgbtrs(const char& trans,
              const int& n,
              const int& kl,
              const int& ku,
              const int& nrhs,
              double* ab,
              const int& ldab,
              int* ipiv,
              double* b,
              const int& ldb,
              int& info);

namespace plask {

/**
 * Symmetric band matrix structure.
 * Data is stored in LAPACK format.
 */
struct DgbMatrix : BandMatrix {
    const size_t shift;  ///< Shift of the diagonal

    aligned_unique_ptr<int> ipiv;

    /**
     * Create matrix
     * \param rank rank of the matrix
     * \param band band size
     */
    DgbMatrix(const Solver* solver, size_t rank, size_t band)
        : BandMatrix(solver, rank, band, ((3 * band + 1 + (15 / sizeof(double))) & ~size_t(15 / sizeof(double))) - 1),
          shift(2 * band) {}

    DgbMatrix(const DgbMatrix&) = delete;

    size_t index(size_t r, size_t c) override {
        assert(r < rank && c < rank);
        if (r < c) {
            assert(c - r <= kd);
            // AB(kl+ku+1+i-j,j) = A(i,j)
            return shift + r + ld * c;
        } else {
            assert(r - c <= kd);
            return shift + c + ld * r;
        }
    }

    void factorize() override {
        solver->writelog(LOG_DETAIL, "Factorizing system");

        int info = 0;
        ipiv.reset(aligned_malloc<int>(rank));

        mirror();

        // Factorize matrix
        dgbtrf(int(rank), int(rank), int(kd), int(kd), data, int(ld + 1), ipiv.get(), info);
        if (info < 0) {
            throw CriticalException("{0}: Argument {1} of `dgbtrf` has illegal value", solver->getId(), -info);
        } else if (info > 0) {
            throw ComputationError(solver->getId(), "Matrix is singular (at {0})", info);
        }
    }

    void solverhs(DataVector<double>& B, DataVector<double>& X) override {
        solver->writelog(LOG_DETAIL, "Solving matrix system");

        int info = 0;
        dgbtrs('N', int(rank), int(kd), int(kd), 1, data, int(ld + 1), ipiv.get(), B.data(), int(B.size()), info);
        if (info < 0) throw CriticalException("{0}: Argument {1} of `dgbtrs` has illegal value", solver->getId(), -info);

        std::swap(B, X);
    }

    /**
     * Multiply matrix by vector
     * \param vector vector to multiply
     * \param result multiplication result
     */
    void mult(const DataVector<const double>& vector, DataVector<double>& result) {
        mirror();
        dgbmv('N', int(rank), int(rank), int(kd), int(kd), 1.0, data, int(ld) + 1, vector.data(), 1, 0.0, result.data(), 1);
    }

    /**
     * Multiply matrix by vector adding the result
     * \param vector vector to multiply
     * \param result multiplication result
     */
    void addmult(const DataVector<const double>& vector, DataVector<double>& result) {
        mirror();
        dgbmv('N', int(rank), int(rank), int(kd), int(kd), 1.0, data, int(ld) + 1, vector.data(), 1, 1.0, result.data(), 1);
    }

  private:
    /// Mirror upper part of the matrix to the lower one
    void mirror() {
        for (size_t i = 0; i < rank; ++i) {
            size_t ldi = shift + (ld + 1) * i;
            size_t knd = min(kd, rank - 1 - i);
            for (size_t j = 1; j <= knd; ++j) data[ldi + j] = data[ldi + ld * j];
        }
    }
};

}  // namespace plask

#endif  // PLASK_COMMON_FEM_GAUSS_MATRIX_H
