#ifndef PLASK__MODULE_THERMAL_BLOCK_MATRIX_H
#define PLASK__MODULE_THERMAL_BLOCK_MATRIX_H

#include <cstddef>
#include <plask/plask.hpp>

#define UPLO 'L'

// BLAS routine to multiply matrix by vector
#define dsbmv F77_GLOBAL(dsbmv,DSBMV)
F77SUB dsbmv(const char& uplo, const int& n, const int& k, const double& alpha, const double* a, const int& lda,
             const double* x, const int& incx, const double& beta, double* y, const int& incy); // y = alpha*A*x + beta*y,

// LAPACK routines to solve set of linear equations
#define dpbtrf F77_GLOBAL(dpbtrf,DPBTRF)
F77SUB dpbtrf(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);

#define dpbtrs F77_GLOBAL(dpbtrs,DPBTRS)
F77SUB dpbtrs(const char& uplo, const int& n, const int& kd, const int& nrhs, double* ab, const int& ldab, double* b, const int& ldb, int& info);


namespace plask { namespace thermal { namespace tstatic {

/**
 * Oversimple symmetric band matrix structure. It only offers easy access to elements and nothing more.
 * Data is stored in LAPACK format.
 */
struct DpbMatrix {

    const size_t size;  ///< Order of the matrix, i.e. number of columns or rows
    const size_t ld;    ///< leading dimension of the matrix
    const size_t kd;    ///< Size of the band reduced by one
    double* data;       ///< Pointer to data

    /**
     * Create matrix
     * \param rank size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y+1])
     */
    DpbMatrix(size_t rank, size_t major):
        size(rank), ld(((major+2+(15/sizeof(double))) & ~size_t(15/sizeof(double))) - 1),
        kd(major+1), data(aligned_malloc<double>(rank*(ld+1))) {}

    /**
     * Create matrix
     * \param rank size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y,z+1])
     * \param minor shift of nodes to the next minor row (mesh[x,y+1,z])
     */
    DpbMatrix(size_t rank, size_t major, size_t minor):
        size(rank), ld(((major+minor+2+(15/sizeof(double))) & ~size_t(15/sizeof(double))) - 1),
        kd(major+minor+1), data(aligned_malloc<double>(rank*(ld+1))) {}


    DpbMatrix(const DpbMatrix&) = delete; // this object is non-copyable

    ~DpbMatrix() { aligned_free(data); }

    /**
     * Return index in data array
     * \param r index of the element row
     * \param c index of the element column
     */
    size_t index(size_t r, size_t c) {
        assert(r < size && c < size);
        if (r < c) {
            assert(c - r <= kd);
//          if UPLO = 'U', AB(kd+i-j,j) = A(i,j) for max(0,j-kd)<=i<=j;
//          if UPLO = 'L', AB(i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
#           if UPLO == 'U'
                return ld * c + r + kd;
#           else
                return ld * r + c;
#           endif
        } else {
            assert(r - c <= kd);
#           if UPLO == 'U'
                return ld * r + c + kd;
#           else
                return ld * c + r;
#           endif
        }
    }

    /**
     * Return reference to array element
     * \param r index of the element row
     * \param c index of the element column
     **/
    double& operator()(size_t r, size_t c) {
        return data[index(r,c)];
    }

    /// Clear the matrix
    void clear() {
        std::fill_n(data, size * (ld+1), 0.);
    }
    
    /**
     * Multiply matrix by vector
     * \param vector vector to multiply
     * \param result multiplication result
     */
    void mult(const DataVector<const double>& vector, DataVector<double>& result) {
        dsbmv(UPLO, size, kd, 1.0, data, ld+1, vector.data(), 1, 0.0, result.data(), 1);
    }

    /**
     * Multiply matrix by vector adding theresult
     * \param vector vector to multiply
     * \param result multiplication result
     */
    void addmult(const DataVector<const double>& vector, DataVector<double>& result) {
        dsbmv(UPLO, size, kd, 1.0, data, ld+1, vector.data(), 1, 1.0, result.data(), 1);
    }
};

}}} // namespace plask::solver::thermal

#endif // PLASK__MODULE_THERMAL_BLOCK_MATRIX_H