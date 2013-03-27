#ifndef PLASK__MODULE_THERMAL_BAND_MATRIX_H
#define PLASK__MODULE_THERMAL_BAND_MATRIX_H

#include <cstddef>
#include <plask/plask.hpp>

#define UPLO 'L'

// LAPACK routines to solve set of linear equations
#define dpbtrf F77_GLOBAL(dpbtrf,DPBTRF)
F77SUB dpbtrf(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);

#define dpbtf2 F77_GLOBAL(dpbtf2,DPBTF2)
F77SUB dpbtf2(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, int& info);

#define dpbtrs F77_GLOBAL(dpbtrs,DPBTRS)
F77SUB dpbtrs(const char& uplo, const int& n, const int& kd, const int& nrhs, double* ab, const int& ldab, double* b, const int& ldb, int& info);

#define dpbequ F77_GLOBAL(dpbequ,DPBEQU)
F77SUB dpbequ(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, double* s, double& scond, double& amax, int& info);

#define dlaqsb F77_GLOBAL(dlaqsb,DLAQSB)
F77SUB dlaqsb(const char& uplo, const int& n, const int& kd, double* ab, const int& ldab, const double* s, const double& scond, const double& amax, char& equed);

namespace plask { namespace solvers { namespace electrical {

/**
 * Oversimple symmetric band matrix structure. It only offers easy access to elements and nothing more.
 * Data is stored in LAPACK format.
 */
struct DpbMatrix {

    const size_t size;  ///< order of the matrix, i.e. number of columns or rows
    const size_t ld;    ///< leading dimension of the matrix reduced by one
    const size_t kd; ///< size of the band reduced by one
    double* data;       ///< pointer to data

    DpbMatrix(size_t rank, size_t bands): size(rank), ld((((bands+1) + (15/sizeof(double))) & ~size_t(15/sizeof(double))) - 1),
                                                          // each column is aligned to 16 bytes
                                         kd(bands), data(aligned_malloc<double>(rank*(ld+1))) {}
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
};

}}} // namespace plask::solver::thermal

#endif // PLASK__MODULE_THERMAL_BAND_MATRIX_H
