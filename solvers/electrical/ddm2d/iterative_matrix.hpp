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
#ifndef PLASK__MODULE_ELECTRICAL_ITERATIVE_MATRIX_H
#define PLASK__MODULE_ELECTRICAL_ITERATIVE_MATRIX_H

#include <algorithm>
#include <plask/plask.hpp>

// Necessary BLAS routines
#define ddot F77_GLOBAL(ddot,DDOT)
F77FUN(double) ddot(const int& n, const double* dx, const int& incx, const double* dy, const int& incy); // return dot(dx,dy)

#define daxpy F77_GLOBAL(daxpy,DAXPY)
F77SUB daxpy(const int& n, const double& sa, const double* sx, const int& incx, double* sy, const int& incy); // sy = sy + sa*sx

#define dsbmv F77_GLOBAL(dsbmv,DSBMV)
F77SUB dsbmv(const char& uplo, const int& n, const int& k, const double& alpha, const double* a, const int& lda,
             const double* x, const int& incx, const double& beta, double* y, const int& incy); // y = alpha*A*x + beta*y,

namespace plask { namespace electrical { namespace drift_diffusion {

/// Error code of solveDCG
struct DCGError: public std::exception {
    const char* msg;
    DCGError(const char* msg): msg(msg) {}
    const char* what() const noexcept override { return msg; }

};

#define LDA 8

struct SparseBandMatrix {
    const ptrdiff_t size;   ///< Order of the matrix, i.e. number of columns or rows
    ptrdiff_t bno[5];      ///< Vector of non-zero band numbers (shift from diagonal)

    double* data;           ///< Data stored in the matrix

    static constexpr size_t kd = 4;
    static constexpr size_t ld = LDA-1;

    /**
     * Create matrix
     * \param size size of the matrix
     * \param major shift of nodes to the next row (mesh[x,y+1])
     */
    SparseBandMatrix(size_t size, size_t major): size(size) {
        bno[0] = 0;  bno[1] = 1;  bno[2] = major - 1;  bno[3] = major;  bno[4] = major + 1;
        data = aligned_malloc<double>(LDA*size);
    }

    ~SparseBandMatrix() {
        aligned_free<double>(data);
    }

    /**
     * Return reference to array element
     * \param r index of the element row
     * \param c index of the element column
     **/
    double& operator()(size_t r, size_t c) {
        if (r < c) std::swap(r, c);
        size_t i = std::find(bno, bno+5, r-c) - bno;
        assert(i != 5);
        return data[LDA*c+i];
    }

    /// Clear the matrix
    void clear() {
        std::fill_n(data, LDA*size, 0.);
    }

    /**
     * Multiplication functor for symmetric banded matrix
     */
    void multiply(double* x, double* y) const { // y = A x
        #pragma omp parallel for
        for (ptrdiff_t r = 0; r < size; ++r) {
            double* datar = data + LDA*r;
            double v = 0.;
            // below diagonal
            for (ptrdiff_t i = 4; i > 0; --i) {
                ptrdiff_t c = r - bno[i];
                if (c >= 0) v += data[LDA*c+i] * x[c];
            }
            // above diagonal
            for (ptrdiff_t i = 0; i < 5; ++i) {
                ptrdiff_t c = r + bno[i];
                if (c < size) v += datar[i] * x[c];
            }
            y[r] = v;
        }
    }

    inline void noUpdate(double*) {}
};


/**
 * Jacobi preconditioner for symmetric banded matrix (i.e. diagonal scaling)
 */
struct PrecondJacobi {

    const SparseBandMatrix& matrix;

    DataVector<double> diag;

    PrecondJacobi(const SparseBandMatrix& A): matrix(A), diag(A.size) {
        for (double *last = matrix.data + A.size*LDA, *m = matrix.data, *d = diag.data(); m < last; m += LDA, ++d)
            *d = 1. / *m;
    }

    void operator()(double* z, double* r) const { // z = inv(M) r
        const double* d = diag.data(), *zend = z + matrix.size-4;
        for (; z < zend; z += 4, r += 4, d += 4) {
            z[0] = r[0] * d[0];
            z[1] = r[1] * d[1];
            z[2] = r[2] * d[2];
            z[3] = r[3] * d[3];
        }
        for (zend += 4; z != zend; ++z, ++r, ++d)
            *z = *r * *d;
    }
};



/**
 * This routine does preconditioned conjugate gradient iteration
 * on the symmetric positive definite system Ax = b.
 *
 * \param[in] matrix Matrix to solve
 * \param[in] msolve (z, r) Functor that solves M z = r for z, given a vector r. Throw exception if an error occures.
 * \param[in,out] x Initial guess of the solution.  If no information on the solution is available then use a zero vector or b.
 * \param[in] b Right hand side.
 * \param[out] err L2 norm of the relative residual error estimate (||b-Ax||/||b||)??.
 *
 *                 This estimate of the true error is not always very accurate.
 * \param[in] eps Requested error tolerence.  System is iterated until ||b-Ax||/||b|| < eps.  Normal choice is 1e-8.
 * \param[in] itmax Maximum number of iterations the user is willing to allow. Default value is 100.
 * \param[in] logfreq Frequency of progress reporting
 * \param[in] log_prefix logging prefix
 * \param[in] updatea Function that updates the matrix A basing on the current solution x
 * \return number of iterations
 * \throw DCGError
 */
template <typename Matrix, typename Preconditioner>
std::size_t solveDCG(Matrix& matrix, const Preconditioner& msolve, double* x, double* b, double& err,
             size_t itmax=10000, double eps=1e-8, size_t logfreq=500, const std::string& log_prefix="",
             void(Matrix::*updatea)(double*)=&Matrix::noUpdate)
{
    DataLog<size_t,double> logger(log_prefix, "conjugate gradient", "iter", "resid");
    size_t logcount = logfreq;

    size_t n = matrix.size;

    aligned_unique_ptr<double[]> r, z, p;
    double bknum, bkden, bk;
    double akden, ak;
    double toobig; // when error estimate gets this big we signal divergence!

    // Calculate norm of right hand size and squared tolerance.
    double bnorm2 = ddot(int(n), b, 1, b, 1);
    double eps2 = eps * eps;

    if (bnorm2 == 0.) {
        std::fill_n(x, n, 0.);
        return 0;
    }

    // Check input data and allocate temporary storage.
    if(n < 2) throw DCGError("system size too small");
    try {
        r.reset(aligned_malloc<double>(n));
        z.reset(aligned_malloc<double>(n));
        p.reset(aligned_malloc<double>(n));
    } catch (...) {
        throw DCGError("could not allocate memory for temporary vectors");
    }

    // Calculate r = b - Ax and initial error.
    matrix.multiply(x, r.get());

    for (std::size_t j = 0; j < n; ++j) r[j] = b[j] - r[j];
    err = ddot(int(n), r.get(), 1, r.get(), 1) / bnorm2;
    if (err < eps2) {
        return 0;
    }
    toobig = err * 1.e8;

    // Iterate!!!
    for (size_t i = 0; i < itmax; i++) {

        // Solve M z = r.
        msolve(z.get(), r.get());

        // Calculate bknum = (z,Mz) and p = z (first iteration).
        if(i == 0) {
            std::copy_n(z.get(), n, p.get());
            bknum = bkden = ddot(int(n), z.get(), 1, r.get(), 1);
        } else {
            // Calculate bknum = (z, r), bkden and bk.
            bknum = ddot(int(n), z.get(), 1, r.get(), 1);
            bk    = bknum / bkden;
            bkden = bknum;

            // Calculate p = z + bk*p
            for (std::size_t j = 0; j < n; ++j)
                p[j] = fma(bk, p[j], z[j]);
        }
        // Calculate z = Ap, akden = (p,Ap) and ak.
        matrix.multiply(p.get(), z.get());

        akden = ddot(int(n), p.get(), 1, z.get(), 1);
        ak    = bknum / akden;

        // Update x and r. Calculate error.
        daxpy(int(n),  ak, p.get(), 1, x, 1);
        daxpy(int(n), -ak, z.get(), 1, r.get(), 1);
        err = ddot(int(n), r.get(), 1, r.get(), 1) / bnorm2;
        if(err < eps2) {
            return i+1;
        }
        if(err > toobig) {
            throw DCGError("divergence of iteration detected");
        }

        if (--logcount == 0) {
            logcount = logfreq;
            logger(i+1, sqrt(err));
        }

        // Update the matrix A
        (matrix.*updatea)(x);
    }
    throw DCGError("iteration limit reached");
    return itmax;
}


}}} // namespace plask::electrical::electrical

#endif // PLASK__MODULE_ELECTRICAL_ITERATIVE_MATRIX_H
