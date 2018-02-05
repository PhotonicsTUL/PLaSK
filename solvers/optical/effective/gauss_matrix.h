#ifndef PLASK__SOLVER__OPTICAL_EFFECTIVE_GAUSS_MATRIX_H
#define PLASK__SOLVER__OPTICAL_EFFECTIVE_GAUSS_MATRIX_H

#include <cstddef>
#include <plask/plask.hpp>
using plask::dcomplex;

// BLAS routine to multiply matrix by vector
#define zgbmv F77_GLOBAL(zgbmv,ZGBMV)
F77SUB zgbmv(const char& trans, const int& m, const int& n, const int& kl, const int& ku, const dcomplex& alpha, dcomplex* a, const int& lda,
             const dcomplex* x, int incx, const dcomplex& beta, dcomplex* y, int incy);


// LAPACK routines to solve set of linear equations
#define zgbtrf F77_GLOBAL(zgbtrf,ZGBTRF)
F77SUB zgbtrf(const int& m, const int& n, const int& kl, const int& ku, dcomplex* ab, const int& ldab, int* ipiv, int& info);

#define zgbtrs F77_GLOBAL(zgbtrs,ZGBTRS)
F77SUB zgbtrs(const char& trans, const int& n, const int& kl, const int& ku, const int& nrhs, dcomplex* ab, const int& ldab, int* ipiv, dcomplex* b, const int& ldb, int& info);

namespace plask { namespace optical { namespace effective {

constexpr const int LD = 7;

/**
 * Oversimple symmetric band matrix structure. It only offers easy access to elements and nothing more.
 * Data is stored in LAPACK format.
 */
struct ZgbMatrix {

    const size_t size;              ///< Order of the matrix, i.e. number of columns or rows
    dcomplex* data;                 ///< Pointer to data

    /**
     * Create matrix
     * \param rank size of the matrix
     */
    ZgbMatrix(size_t rank): size(rank), data(aligned_malloc<dcomplex>(LD*rank)) {}

    ZgbMatrix(const ZgbMatrix&) = delete; // this object is non-copyable

    ~ZgbMatrix() { aligned_free(data); }

    /**
     * Return index in data array
     * \param r index of the element row
     * \param c index of the element column
     */
    size_t index(size_t r, size_t c) {
        assert(r < size && c < size);
        assert(abs(int(c)-int(r)) < 3);
        // AB(kl+ku+1+i-j,j) = A(i,j)
        return (LD-1)*c + r + 4;
    }

    /**
     * Return reference to array element
     * \param r index of the element row
     * \param c index of the element column
     **/
    dcomplex& operator()(size_t r, size_t c) {
        return data[index(r,c)];
    }

    /// Clear the matrix
    void clear() {
        std::fill_n(data, LD*size, 0.);
    }

    /**
     * Multiply matrix by vector
     * \param vector vector to multiply
     * \param result multiplication result
     */
    void mult(const DataVector<const dcomplex>& vector, DataVector<dcomplex>& result) {
        zgbmv('N', int(size), int(size), 2, 2, 1., data, LD, vector.data(), 1, 0., result.data(), 1);
    }

    /**
     * Multiply matrix by vector adding the result
     * \param vector vector to multiply
     * \param result multiplication result
     */
    void addmult(const DataVector<const dcomplex>& vector, DataVector<dcomplex>& result) {
        zgbmv('N', int(size), int(size), 2, 2, 1., data, LD, vector.data(), 1, 1., result.data(), 1);
    }

    /// Compute matrix determinant
    dcomplex determinant() {
        int info = 0;
        aligned_unique_ptr<int> upiv(aligned_malloc<int>(size));
        int* ipiv = upiv.get();
        zgbtrf(int(size), int(size), 2, 2, data, LD, ipiv, info);
        assert(info >= 0);

        dcomplex det = 1.;
        for (std::size_t i = 0; i < size; ++i) {
            det *= data[LD*i + 4];
            if (ipiv[i] != int(i+1)) det = -det;
        }
        return det;
    }
};

}}} // # namespace plask::gain::freecarrier

#undef LD

#endif // PLASK__SOLVER__OPTICAL_EFFECTIVE_GAUSS_MATRIX_H

