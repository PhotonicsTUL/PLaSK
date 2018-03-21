#ifndef PLASK__MODULE_ELECTRICAL_STATIC_CONJUGATE_GRADIENT_H
#define PLASK__MODULE_ELECTRICAL_STATIC_CONJUGATE_GRADIENT_H

#include <algorithm>
#include <plask/plask.hpp>

// Necessary BLAS routines
#define ddot F77_GLOBAL(ddot,DDOT)
F77FUN(double) ddot(const int& n, const double* dx, const int& incx, const double* dy, const int& incy); // return dot(dx,dy)

#define daxpy F77_GLOBAL(daxpy,DAXPY)
F77SUB daxpy(const int& n, const double& sa, const double* sx, const int& incx, double* sy, const int& incy); // sy = sy + sa*sx

namespace plask { namespace electrical { namespace shockley {

/// Error code of solveDCG
struct DCGError: public std::exception {
    const char* msg;
    DCGError(const char* msg): msg(msg) {}
    const char* what() const noexcept override { return msg; }

};


/**
 * This routine does preconditioned conjugate gradient iteration
 * on the symmetric positive definite system Ax = b.
 *
 * \param[in] matrix Matrix to solve
 * \param[in] msolve (z, r) Functor that solves M z = r for z, given a vector r. Throw exception if an error occures.
 * \param[in,out] x Initial guess of the solution.  If no information on the solution is available then use a zero vector or b.
 * \param[in] b Right hand side.
 * \param[out] err L2 norm of the relative residual error estimate (||b-Ax||/||b||)².
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
    Data2DLog<size_t,double> logger(log_prefix, "conjugate gradient", "iter", "resid");
    size_t logcount = logfreq;

    size_t n = matrix.size;

    aligned_unique_ptr<double[]> r = nullptr, z = nullptr, p = nullptr;
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


}}} // namespaces

#endif // PLASK__MODULE_ELECTRICAL_STATIC_CONJUGATE_GRADIENT_H
