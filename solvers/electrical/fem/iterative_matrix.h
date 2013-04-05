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

namespace plask { namespace solvers { namespace electrical {

/// Error code of solveDCG
struct DCGError: public std::exception {
    const char* msg;
    DCGError(const char* msg): msg(msg) {}
    const char* what() const noexcept { return msg; }

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
     * \param rank size of the matrix
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
            register double v = 0.;
            // below diagonal
            for (register ptrdiff_t i = 4; i > 0; --i) {
                register ptrdiff_t c = r - bno[i];
                if (c >= 0) v += data[LDA*c+i] * x[c];
            }
            // above diagonal
            for (register ptrdiff_t i = 0; i < 5; ++i) {
                register ptrdiff_t c = r + bno[i];
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

    PrecondJacobi(const SparseBandMatrix& A): matrix(A) {}

    void operator()(double* z, double* r) const { // z = inv(M) r
        double* m = matrix.data, *zend = z + matrix.size-4;
        for (; z < zend; z += 4, r += 4, m += 4*LDA) {
            z[0] = r[0] / m[0];
            z[1] = r[1] / m[LDA];
            z[2] = r[2] / m[2*LDA];
            z[3] = r[3] / m[3*LDA];
        }
        for (zend += 4; z != zend; ++z, ++r, m += LDA)
            *z = *r / *m;
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
int solveDCG(Matrix& matrix, const Preconditioner& msolve, double* x, double* b, double& err,
             size_t itmax=10000, double eps=1e-8, size_t logfreq=500, const std::string& log_prefix="",
             void(Matrix::*updatea)(double*)=&Matrix::noUpdate)
{
    Data2DLog<size_t,double> logger(log_prefix, "conjugate gradient", "iter", "resid");
    size_t logcount = logfreq;

    size_t n = matrix.size;

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
        r = aligned_malloc<double>(n);
        z = aligned_malloc<double>(n);
        p = aligned_malloc<double>(n);
    } catch (...) {
        aligned_free(p); aligned_free(z); aligned_free(r);
        throw DCGError("could not allocate memory for temporary vectors");
    }

    // Calculate r = b - Ax and initial error.
    try {
        matrix.multiply(x, r);
    } catch (...) {
        aligned_free(p); aligned_free(z); aligned_free(r);
        throw;
    }
    for (register int j = 0; j < n; ++j) r[j] = b[j] - r[j];
    err = ddot(n, r, 1, r, 1) / bnorm2;
    if (err < eps2) {
        aligned_free(p); aligned_free(z); aligned_free(r);
        return 0;
    }
    toobig = err * 1.e8;

    // Iterate!!!
    for (register size_t i = 0; i < itmax; i++) {

        // Solve M z = r.
        try {
            msolve(z, r);
        } catch (...) {
            aligned_free(p); aligned_free(z); aligned_free(r);
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

            // Calculate p = z + bk*p
            for (register int j = 0; j < n; ++j)
                p[j] = fma(bk, p[j], z[j]);
        }
        // Calculate z = Ap, akden = (p,Ap) and ak.
        try {
            matrix.multiply(p, z);
        } catch (...) {
            aligned_free(p); aligned_free(z); aligned_free(r);
            throw;
        }
        akden = ddot(n, p, 1, z, 1);
        ak    = bknum / akden;

        // Update x and r. Calculate error.
        daxpy(n,  ak, p, 1, x, 1);
        daxpy(n, -ak, z, 1, r, 1);
        err = ddot(n, r, 1, r, 1) / bnorm2;
        if(err < eps2) {
            aligned_free(p); aligned_free(z); aligned_free(r);
            return i+1;
        }
        if(err > toobig) {
            aligned_free(p); aligned_free(z); aligned_free(r);
            throw DCGError("divergence of iteration detected");
        }

        if (--logcount == 0) {
            logcount = logfreq;
            logger(i+1, sqrt(err));
        }

        // Update the matrix A
        (matrix.*updatea)(x);
    }
    aligned_free(p); aligned_free(z); aligned_free(r);
    throw DCGError("iteration limit reached");
    return itmax;
}


}}} // namespace plask::solvers::electrical

#endif // PLASK__MODULE_ELECTRICAL_ITERATIVE_MATRIX_H
