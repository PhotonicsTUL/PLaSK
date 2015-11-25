#ifndef PLASK__SOLVER__GAIN_FREECARRIER_GAUSS_MATRIX_H
#define PLASK__SOLVER__GAIN_FREECARRIER_GAUSS_MATRIX_H

#include <cstddef>
#include <plask/plask.hpp>

// BLAS routine to multiply matrix by vector
#define dgbmv F77_GLOBAL(dgbmv,DGBMV)
F77SUB dgbmv(const char& trans, const int& m, const int& n, const int& kl, const int& ku, const double& alpha, double* a, const int& lda,
             const double* x, int incx, const double& beta, double* y, int incy);


// LAPACK routines to solve set of linear equations
#define dgbtrf F77_GLOBAL(dgbtrf,DGBTRF)
F77SUB dgbtrf(const int& m, const int& n, const int& kl, const int& ku, double* ab, const int& ldab, int* ipiv, int& info);

#define dgbtrs F77_GLOBAL(dgbtrs,DGBTRS)
F77SUB dgbtrs(const char& trans, const int& n, const int& kl, const int& ku, const int& nrhs, double* ab, const int& ldab, int* ipiv, double* b, const int& ldb, int& info);

#define LD 7

namespace plask { namespace gain { namespace freecarrier {

/**
 * Oversimple symmetric band matrix structure. It only offers easy access to elements and nothing more.
 * Data is stored in LAPACK format.
 */
struct DgbMatrix {

    const size_t size;              ///< Order of the matrix, i.e. number of columns or rows
    double* data;                   ///< Pointer to data

    aligned_unique_ptr<int> ipiv;

    /**
     * Create matrix
     * \param rank size of the matrix
     */
    DgbMatrix(size_t rank): size(rank), data(aligned_malloc<double>(LD*rank)) {}

    DgbMatrix(const DgbMatrix&) = delete; // this object is non-copyable

    ~DgbMatrix() { aligned_free(data); }

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
    double& operator()(size_t r, size_t c) {
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
    void mult(const DataVector<const double>& vector, DataVector<double>& result) {
        dgbmv('N', size, size, 2, 2, 1., data, LD, vector.data(), 1, 0., result.data(), 1);
    }

    /**
     * Multiply matrix by vector adding the result
     * \param vector vector to multiply
     * \param result multiplication result
     */
    void addmult(const DataVector<const double>& vector, DataVector<double>& result) {
        dgbmv('N', size, size, 2, 2, 1., data, LD, vector.data(), 1, 1., result.data(), 1);
    }

    /// Compute matrix determinant
    double determinant() {
        int info = 0;
        aligned_unique_ptr<int> upiv(aligned_malloc<int>(size));
        int* ipiv = upiv.get();
        dgbtrf(size, size, 2, 2, data, LD, ipiv, info);
        assert(info >= 0);

        double det = 1.;
        for (int i = 0; i < size; ++i) {
            det *= data[LD*i + 4];
            if (ipiv[i] != i+1) det = -det;
        }
        return det;
    }
};

}}} // # namespace plask::gain::freecarrier

#undef LD

#endif // PLASK__SOLVER__GAIN_FREECARRIER_GAUSS_MATRIX_H

