#ifndef PLASK__SOLVER__ELECTR3D_ITERATIVE_MATRIX_H
#define PLASK__SOLVER__ELECTR3D_ITERATIVE_MATRIX_H

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

namespace plask { namespace solvers { namespace electrical3d {

/// Error code of solveDCG
struct DCGError: public std::exception {
    const char* msg;
    DCGError(const char* msg): msg(msg) {}
    const char* what() const noexcept override { return msg; }

};

#define LDA 16

struct SparseBandMatrix {
    const ptrdiff_t size;   ///< Order of the matrix, i.e. number of columns or rows
    ptrdiff_t bno[14];      ///< Vector of non-zero band numbers (shift from diagonal)

    double* data;           ///< Data stored in the matrix

    static constexpr size_t kd = 13;
    static constexpr size_t ld = LDA-1;

    /**
     * Create matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y,z+1])
     * \param minor shift of nodes to the next minor row (mesh[x,y+1,z])
     */
    SparseBandMatrix(size_t size, size_t major, size_t minor): size(size) {
                                      bno[0]  =             0;  bno[1]  =                 1;
        bno[2]  =         minor - 1;  bno[3]  =         minor;  bno[4]  =         minor + 1;
        bno[5]  = major - minor - 1;  bno[6]  = major - minor;  bno[7]  = major - minor + 1;
        bno[8]  = major         - 1;  bno[9]  = major        ;  bno[10] = major         + 1;
        bno[11] = major + minor - 1;  bno[12] = major + minor;  bno[13] = major + minor + 1;

        data = aligned_malloc<double>(LDA*size);
    }

    ~SparseBandMatrix() {
        aligned_free<double>(data);
    }

    /**
     * Return reference to array element.
     * @param r index of the element row
     * @param c index of the element column
     * @return reference to array element
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
            double v = 0.;
            // below diagonal
            for (ptrdiff_t i = 13; i > 0; --i) {
                ptrdiff_t c = r - bno[i];
                if (c >= 0) v += data[LDA*c+i] * x[c];
            }
            // above diagonal
            for (ptrdiff_t i = 0; i < 14; ++i) {
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
            // *reinterpret_cast<v2double*>(z) = *reinterpret_cast<const v2double*>(r) * *reinterpret_cast<const v2double*>(d);
            // *reinterpret_cast<v2double*>(z+2) = *reinterpret_cast<const v2double*>(r+2) * *reinterpret_cast<const v2double*>(d+2);
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

    aligned_unique_ptr<double[]> r, z, p;
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
        r.reset(aligned_malloc<double>(n));
        z.reset(aligned_malloc<double>(n));
        p.reset(aligned_malloc<double>(n));
    } catch (...) {
        throw DCGError("could not allocate memory for temporary vectors");
    }

    // Calculate r = b - Ax and initial error.
    matrix.multiply(x, r.get());

    for (int j = 0; j < n; ++j) r[j] = b[j] - r[j];
    err = ddot(n, r.get(), 1, r.get(), 1) / bnorm2;
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
            bknum = bkden = ddot(n, z.get(), 1, r.get(), 1);
        } else {
            // Calculate bknum = (z, r), bkden and bk.
            bknum = ddot(n, z.get(), 1, r.get(), 1);
            bk    = bknum / bkden;
            bkden = bknum;

            // Calculate p = z + bk*p
            for (int j = 0; j < n; ++j)
                p[j] = fma(bk, p[j], z[j]);
        }
        // Calculate z = Ap, akden = (p,Ap) and ak.
        matrix.multiply(p.get(), z.get());

        akden = ddot(n, p.get(), 1, z.get(), 1);
        ak    = bknum / akden;

        // Update x and r. Calculate error.
        daxpy(n,  ak, p.get(), 1, x, 1);
        daxpy(n, -ak, z.get(), 1, r.get(), 1);
        err = ddot(n, r.get(), 1, r.get(), 1) / bnorm2;
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


}}} // namespace plask::solvers::electrical3d

#endif // PLASK__SOLVER__ELECTR3D_ITERATIVE_MATRIX_H
