#ifndef PLASK__MODULE_THERMAL_ITERATIVE_MATRIX_H
#define PLASK__MODULE_THERMAL_ITERATIVE_MATRIX_H

#include <algorithm>
#include <plask/config.h>

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

#define LDA 16

struct SparseBandMatrix {
    const ptrdiff_t size;   ///< Order of the matrix, i.e. number of columns or rows
    ptrdiff_t bno[14];      ///< Vector of non-zero band numbers (shift from diagonal)

    double* data;           ///< Data stored in the matrix

    static const size_t bands;

    /**
     * Create matrix
     * \param rank size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y,z+1])
     * \param minor shift of nodes to the next minor row (mesh[x,y+1,z])
     */
    SparseBandMatrix(size_t size, size_t major, size_t minor): size(size) {
                                      bno[0]  =             0;  bno[1]  =                 1;
        bno[2]  =         minor - 1;  bno[3]  =         minor;  bno[4]  =         minor + 1;
        bno[5]  = major - minor - 1;  bno[6]  = major - minor;  bno[7]  = major - minor + 1;
        bno[8]  = major         - 1;  bno[9]  = major        ;  bno[10] = major         + 1;
        bno[11] = major + minor - 1;  bno[12] = major + minor;  bno[13] = major + minor + 1;

        data = new double[LDA*size];
    }

    ~SparseBandMatrix() {
        delete[] data;
    }

    /**
     * Return reference to array element
     * \param r index of the element row
     * \param c index of the element column
     **/
    double& operator()(size_t r, size_t c) {
        if (r < c) std::swap(r, c);
        size_t i = std::find(bno, bno+14, r-c) - bno;
        assert(i != 14);
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
            for (register ptrdiff_t i = 13; i > 0; --i) {
                register ptrdiff_t c = r - bno[i];
                if (c >= 0) v += data[LDA*c+i] * x[c];
            }
            // above diagonal
            for (register ptrdiff_t i = 0; i < 14; ++i) {
                register ptrdiff_t c = r + bno[i];
                if (c < size) v += datar[i] * x[c];
            }
            y[r] = v;
        }
    }


    /**
     * Jacobi preconditioner for symmetric banded matrix (i.e. diagonal scaling)
     */
    void precondJacobi(double* z, double* r) const { // z = inv(M) r
        for (double* m = data, *zend = z + size; z < zend; ++z, ++r, m += LDA)
            *z = *r / *m;
    }

    inline void noUpdate(double*) {}

};



/**
 * This routine does preconditioned conjugate gradient iteration
 * on the symmetric positive definte system Ax = b.
 *
 * \param[in] matrix Matrix to solve
 * \param[in] msolve (z, r) Functor that solves M z = r for z, given a vector r. Throw exception if an error occures.
 * \param[in,out] x Initial guess of the solution.  If no information on the solution is available then use a zero vector or b.
 * \param[in] b Right hand side.
 * \param[out] err L2 norm of the relative residual error estimate (||b-Ax||/||b||)².
 *
 *                 This estimate of the true error is not always very accurate.
 * \param[in] eps Requested error tollerence.  System is iterated until ||b-Ax||/||b|| < eps.  Normal choice is 1e-8.
 * \param[in] itmax Maximum number of iterations the user is willing to allow. Default value is 100.
 * \param[in] updatea Function that updates the matrix A basing on the current solution x
 * \return number of iterations
 * \throw DCGError
 */
template <typename Matrix>
int solveDCG(Matrix& matrix, void(Matrix::*msolve)(double*,double*)const, double* x, double* b, double& err,
             int itmax=10000, double eps=1e-8, void(Matrix::*updatea)(double*)=&Matrix::noUpdate)
{
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
        r = new double[n];
        z = new double[n];
        p = new double[n];
    } catch (...) {
        delete[] p; delete[] z; delete[] r;
        throw DCGError("could not allocate memory for temporary vectors");
    }

    // Calculate r = b - Ax and initial error.
    try {
        matrix.multiply(x, r);
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
            (matrix.*msolve)(z, r);
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
            matrix.multiply(p, z);
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
        (matrix.*updatea)(x);
    }
    delete[] p; delete[] z; delete[] r;
    throw DCGError("iteration limit reached");
    return itmax;
}


}}} // namespace plask::solvers::thermal

#endif // PLASK__MODULE_THERMAL_ITERATIVE_MATRIX_H
