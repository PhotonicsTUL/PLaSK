#ifndef PLASK__MODULE_ELECTRICAL_GAUSS_MATRIX_H
#define PLASK__MODULE_ELECTRICAL_GAUSS_MATRIX_H

#include <cstddef>
#include <plask/plask.hpp>

// LAPACK routines to solve set of linear equations
#define dgbtrf F77_GLOBAL(dgbtrf,DGBTRF)
F77SUB dgbtrf(const int& m, const int& n, const int& kl, const int& ku, double* ab, const int& ldab, int* ipiv, int& info);

#define dgbtrs F77_GLOBAL(dgbtrs,DGBTRS)
F77SUB dgbtrs(const char& trans, const int& n, const int& kl, const int& ku, const int& nrhs, double* ab, const int& ldab, int* ipiv, double* b, const int& ldb, int& info);


namespace plask { namespace solvers { namespace electrical {

/**
 * Oversimple symmetric band matrix structure. It only offers easy access to elements and nothing more.
 * Data is stored in LAPACK format.
 */
struct DgbMatrix {

    const size_t size;  ///< Order of the matrix, i.e. number of columns or rows
    const size_t ld;    ///< leading dimension of the matrix
    const size_t kd;    ///< Size of the band reduced by one
    const size_t shift; ///< Shift of the diagonal
    double* data;       ///< Pointer to data

    /**
     * Create matrix
     * \param rank size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y+1])
     */
    DgbMatrix(size_t rank, size_t major):
        size(rank), ld(((3*major+4+(15/sizeof(double))) & ~size_t(15/sizeof(double))) - 1),
        kd(major+1), shift(2*major+2), data(aligned_malloc<double>(rank*(ld+1))) {}

    DgbMatrix(const DgbMatrix&) = delete; // this object is non-copyable

    ~DgbMatrix() { aligned_free(data); }

    /**
     * Return index in data array
     * \param r index of the element row
     * \param c index of the element column
     */
    size_t index(size_t r, size_t c) {
        assert(r < size && c < size);
        if (r < c) {
            assert(c - r <= kd);
            // AB(kl+ku+1+i-j,j) = A(i,j)
            return shift + r + ld*c;
        } else {
            assert(r - c <= kd);
            return shift + c + ld*r;
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

    /// Mirror upper part of the matrix to the lower one
    void mirror() {
        for (size_t i = 0; i < size; ++i) {
            size_t ldi = shift + (ld+1) * i;
            size_t knd = min(kd, size-1-i);
            for (size_t j = 1; j <= knd; ++j)
                data[ldi + j] = data[ldi + ld * j];
        }
    }
};

}}} // namespace plask::solver::electrical

#endif // PLASK__MODULE_ELECTRICAL_GAUSS_MATRIX_H
