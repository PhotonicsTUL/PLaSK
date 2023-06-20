/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2023 Lodz University of Technology
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
#ifndef PLASK_COMMON_FEM_MATRIX_HPP
#define PLASK_COMMON_FEM_MATRIX_HPP

#include <plask/plask.hpp>

namespace plask {

struct FemMatrix {
    const size_t size;     ///< Order of the matrix, i.e. number of columns or rows
    const size_t ld;       ///< leading dimension of the matrix
    const size_t kd;       ///< Size of the band reduced by one
    double* data;          ///< Pointer to data
    const Solver* solver;  ///< Solver owning the matrix

    FemMatrix(const Solver* solver, size_t size, size_t kd, size_t ld)
        : size(size), ld(ld), kd(kd), data(aligned_malloc<double>(size * (ld + 1))), solver(solver) {
        clear();
    }

    FemMatrix(const FemMatrix&) = delete;

    virtual ~FemMatrix() { aligned_free<double>(data); }

    /**
     * Return reference to array element
     * \param r index of the element row
     * \param c index of the element column
     **/
    double& operator()(size_t r, size_t c) { return data[index(r, c)]; }

    /**
     * Return index in data array
     * \param r index of the element row
     * \param c index of the element column
     */
    virtual size_t index(size_t r, size_t c) = 0;

    /// Clear the matrix
    void clear() {
        std::fill_n(data, size * (ld + 1), 0.);
    }

    template <typename BoundaryConditonsT> void applyBC(const BoundaryConditonsT& bconds, DataVector<double>& B) {
        // boundary conditions of the first kind
        for (auto cond : bconds) {
            for (auto r : cond.place) {
                setBC(B, r, cond.value);
            }
        }
    }

    /**
     * Factorize the matrix in advance to speed up the solution
     * \param solver solver to use
     */
    virtual void factorize() {}

    /**
     * Solve for the right-hand-side of a system of linear equations
     * \param solver solver to use
     * \param[inout] B right hand side of the equation, on output may be interchanged with X
     * \param[inout] X initial estimate of the solution, on output contains the solution (may be interchanged with B)
     * \return number of iterations
     */
    virtual void solverhs(DataVector<double>& B, DataVector<double>& X) = 0;

    /**
     * Solve the set of linear equations
     * \param solver solver to use
     * \param[inout] B right hand side of the equation, on output may be interchanged with X
     * \param[inout] X initial estimate of the solution, on output contains the solution (may be interchanged with B)
     */
    void solve(DataVector<double>& B, DataVector<double>& X) {
        factorize();
        solverhs(B, X);
    }

    /**
     * Solve the set of linear equations
     * \param solver solver to use
     * \param[inout] B right hand side of the equation, on output contains the solution
     * \return number of iterations
     */
    void solve(DataVector<double>& B) {
        solve(B, B);
    }

    /**
     * Multiply matrix by vector
     * \param vector vector to multiply
     * \param result multiplication result
     */
    virtual void mult(const DataVector<const double>& vector, DataVector<double>& result) = 0;

    /**
     * Multiply matrix by vector adding theresult
     * \param vector vector to multiply
     * \param result multiplication result
     */
    virtual void addmult(const DataVector<const double>& vector, DataVector<double>& result) = 0;

  protected:
    virtual void setBC(DataVector<double>& B, size_t r, double val) {
        B[r] = val;
        (*this)(r, r) = 1.;
        size_t start = (r > kd) ? r - kd : 0;
        size_t end = (r + kd < size) ? r + kd + 1 : size;
        for (size_t c = start; c < r; ++c) {
            B[c] -= (*this)(r, c) * val;
            (*this)(r, c) = 0.;
        }
        for (size_t c = r + 1; c < end; ++c) {
            B[c] -= (*this)(r, c) * val;
            (*this)(r, c) = 0.;
        }
    }
};

}  // namespace plask

#endif
