#ifndef PLASK__MODULE_THERMAL_DCG_H
#define PLASK__MODULE_THERMAL_DCG_H

#include <algorithm>
#include <plask/config.h>

#include "band_matrix.h"

// Necessary BLAS routines
#define ddot F77_GLOBAL(ddot,DDOT)
F77FUN(double) ddot(const int& n, const double* dx, const int& incx, const double* dy, const int& incy); // return dot(dx,dy)

#define daxpy F77_GLOBAL(daxpy,DAXPY)
F77SUB daxpy(const int& n, const double& sa, const double* sx, const int& incx, double* sy, const int& incy); // sy = sy + sa*sx

#define dsbmv F77_GLOBAL(dsbmv,DSBMV)
F77SUB dsbmv(const char& uplo, const int& n, const int& k, const double& alpha, const double* a, const int& lda,
             const double* x, const int& incx, const double& beta, double* y, const int& incy); // y = alpha*A*x + beta*y,

namespace plask { namespace solvers { namespace thermal3d {

/// Error code of solveDCG
struct DCGError: public std::exception {
    const char* msg;
    DCGError(const char* msg): msg(msg) {}
    const char* what() const noexcept { return msg; }

};

static inline void noUpdate(double*) {}

/**
 * This routine does preconditioned conjugate gradient iteration
 * on the symmetric positive definte system Ax = b.
 *
 * \param[in] n Size of system to be iterated (ie A is nxn).
 * \param[in] atimes (x, y) Functor that calculates y = A x, given an x vector. Throw exception if an error occures.
 * \param[in] msolve (z, r) Functor that solves M z = r for z, given a vector r. Throw exception if an error occures.
 * \param[in,out] x Initial guess of the solution.  If no information on the solution is available then use a zero vector or b.
 * \param[in] b Right hand side.
 * \param[out] err L2 norm of the relative residual error estimate (||b-Ax||/||b||)².
 *
 *                 This estimate of the true error is not always very accurate.
 * \param[in] eps Requested error tollerence.  System is iterated until ||b-Ax||/||b|| < eps.  Normal choice is 1e-18.
 * \param[in] itmax Maximum number of iterations the user is willing to allow. Default value is 100.
 * \param[in] updatea Function that updates the matrix A basing on the current solution x
 * \return number of iterations
 * \throw DCGError
 */
template <typename AFun, typename MFun>
int solveDCG(int n, AFun atimes, MFun msolve, double* x, double* b, double& err, int itmax=2000, double eps=1e-18, std::function<void(double*)>updatea=noUpdate)
{
    double *r = nullptr, *z = nullptr, *p = nullptr;
    double bknum, bkden, bk;
    double akden, ak;
    double toobig; // when error estimate gets this big we signal divergence!

    // Calculate norm of right hand size and squared tolerance.
    double bnorm2 = ddot(n, b, 1, b, 1);
    double eps2 = eps * eps;

    if (bnorm2 == 0.) {
        std::fill_n(x, n, 0.);
        return 0;
    }

    // Check input data and allocate temporary storage.
    if(n < 2) throw DCGError("system size too small");
    try {
        r = new double[n];
        z = new double[n];
        p = new double[n];
    } catch (...) {
        delete[] p; delete[] z; delete[] r;
        throw DCGError("could not allocate memory for temporary vectors");
    }

    // Calculate r = b - Ax and initial error.
    try {
        atimes(x, r);
    } catch (...) {
        delete[] p; delete[] z; delete[] r;
        throw;
    }
    for (register int j = 0; j < n; ++j) r[j] = b[j] - r[j];
    err = ddot(n, r, 1, r, 1) / bnorm2;
    if (err < eps2) {
        delete[] p; delete[] z; delete[] r;
        return 0;
    }
    toobig = err * 1.e8;

    // Iterate!!!
    for (register int i = 0; i < itmax; i++) {

        // Solve M z = r.
        try {
            msolve(z, r);
        } catch (...) {
            delete[] p; delete[] z; delete[] r;
            throw;
        }

        // Calculate bknum = (z,Mz) and p = z (first iteration).
        if(i == 0) {
            std::copy_n(z, n, p);
            bknum = bkden = ddot(n, z, 1, r, 1);
        } else {
            // Calculate bknum = (z, r), bkden and bk.
            bknum = ddot(n, z, 1, r, 1);
            bk    = bknum / bkden;
            bkden = bknum;

            // Calculate p = z + bk*p,If we want
            for (register int j = 0; j < n; ++j)
                p[j] = z[j] + bk*p[j];
        }
        // Calculate z = Ap, akden = (p,Ap) and ak.
        try {
            atimes(p, z);
        } catch (...) {
            delete[] p; delete[] z; delete[] r;
            throw;
        }
        akden = ddot(n, p, 1, z, 1);
        ak    = bknum / akden;

        // Update x and r. Calculate error.
        daxpy(n,  ak, p, 1, x, 1);
        daxpy(n, -ak, z, 1, r, 1);
        err = ddot(n, r, 1, r, 1) / bnorm2;
        if(err < eps2) {
            delete[] p; delete[] z; delete[] r;
            return i+1;
        }
        if(err > toobig) {
            delete[] p; delete[] z; delete[] r;
            throw DCGError("divergence of iteration detected");
        }

        // Update the matrix A
        updatea(x);
    }
    delete[] p; delete[] z; delete[] r;
    throw DCGError("iteration limit reached");
    return itmax;
}


/**
 * Multiplication functor for symmetric banded matrix
 */
struct Atimes {

    const SparseBandMatrix& matrix;

    Atimes(const SparseBandMatrix& matrix): matrix(matrix) {}

    void operator()(double* x, double* y) { // y = A x
        ptrdiff_t n = matrix.size;
        #pragma omp parallel for
        for (size_t r = 0; r < n; ++r) {
            y[r] = 0.;
            // below diagonal
            for (size_t j = 13; j > 0; --j) {
                ptrdiff_t c = r - matrix.bno[j];
                assert(c < n);
                if (c >= 0) y[r] += matrix.data[j][c] * x[c];
            }
            // above diagonal
            for (size_t j = 0; j < 14; ++j) {
                ptrdiff_t c = r + matrix.bno[j];
                assert(c >= 0);
                if (c < n) y[r] += matrix.data[j][r] * x[c];
            }
        }
    }
};

/**
 * Jacobi preconditioner for symmetric banded matrix (i.e. diagonal scaling)
 */
struct MsolveJacobi {

    const SparseBandMatrix& matrix;

    MsolveJacobi(const SparseBandMatrix& matrix): matrix(matrix) {}

    void operator()(double* z, double* r) { // z = inv(M) r
        double* zend = z + matrix.size;
        for (double* m = matrix.data[0]; z < zend; ++z, ++r, ++m) {
            *z = *r / *m;
        }
    }

};

}}} // namespace plask::solvers::thermal

#endif // PLASK__MODULE_THERMAL_DCG_H
